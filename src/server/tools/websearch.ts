import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "@earendil-works/pi-ai";
import { Readability } from "@mozilla/readability";
import { JSDOM } from "jsdom";
import TurndownService from "turndown";
import { gfm } from "turndown-plugin-gfm";

const braveApiKey = process.env.BRAVE_API_KEY;

const webSearchParameters = Type.Object({
	query: Type.String({ description: "Search query." }),
	count: Type.Optional(Type.Number({ description: "Number of search results. Defaults to 5, maximum 20." })),
	country: Type.Optional(Type.String({ description: "Two-letter country code. Defaults to DE." })),
	freshness: Type.Optional(
		Type.String({ description: "Optional freshness filter: pd, pw, pm, py, or YYYY-MM-DDtoYYYY-MM-DD." }),
	),
});

const pageContentParameters = Type.Object({
	url: Type.String({ description: "Absolute URL of the page to fetch and extract as readable markdown." }),
});

interface BraveSearchResult {
	title?: string;
	url?: string;
	description?: string;
	age?: string;
	page_age?: string;
}

interface BraveSearchResponse {
	web?: {
		results?: BraveSearchResult[];
	};
}

interface WebSearchResultDetails {
	query: string;
	count: number;
	country: string;
	freshness?: string;
	results: Array<{ title: string; url: string; snippet: string; age?: string }>;
}

interface PageContentDetails {
	url: string;
	chars: number;
}

function htmlToMarkdown(html: string): string {
	const turndown = new TurndownService({ headingStyle: "atx", codeBlockStyle: "fenced" });
	turndown.use(gfm);
	turndown.addRule("removeEmptyLinks", {
		filter: (node) => node.nodeName === "A" && !node.textContent?.trim(),
		replacement: () => "",
	});
	return turndown
		.turndown(html)
		.replace(/\[\\?\[\s*\\?\]\]\([^)]*\)/g, "")
		.replace(/ +/g, " ")
		.replace(/\s+,/g, ",")
		.replace(/\s+\./g, ".")
		.replace(/\n{3,}/g, "\n\n")
		.trim();
}

function formatSearchResults(details: WebSearchResultDetails): string {
	if (details.results.length === 0) return `Keine Suchergebnisse für: ${details.query}`;
	return details.results
		.map((result, index) => {
			const age = result.age ? `\nAge: ${result.age}` : "";
			return `--- Result ${index + 1} ---\nTitle: ${result.title}\nLink: ${result.url}${age}\nSnippet: ${result.snippet}`;
		})
		.join("\n\n");
}

async function searchWebWithBrave(
	query: string,
	count: number,
	country: string,
	freshness?: string,
): Promise<WebSearchResultDetails> {
	if (!braveApiKey) throw new Error("BRAVE_API_KEY is not set");
	const resultCount = Math.max(1, Math.min(20, Math.floor(count)));
	const normalizedCountry = country.trim().toUpperCase().slice(0, 2) || "DE";
	const url = new URL("https://api.search.brave.com/res/v1/web/search");
	url.searchParams.set("q", query);
	url.searchParams.set("count", String(resultCount));
	url.searchParams.set("country", normalizedCountry);
	url.searchParams.set("search_lang", "de");
	url.searchParams.set("ui_lang", "de-DE");
	if (freshness) url.searchParams.set("freshness", freshness);
	const response = await fetch(url, {
		headers: {
			accept: "application/json",
			"accept-encoding": "gzip",
			"x-subscription-token": braveApiKey,
		},
	});
	if (!response.ok) {
		const body = await response.text().catch(() => "");
		throw new Error(`Brave Search failed: HTTP ${response.status}${body ? ` ${body.slice(0, 300)}` : ""}`);
	}
	const data = (await response.json()) as BraveSearchResponse;
	const results = (data.web?.results ?? []).slice(0, resultCount).map((entry) => ({
		title: entry.title ?? "Untitled",
		url: entry.url ?? "",
		snippet: entry.description ?? "",
		age: entry.age ?? entry.page_age,
	}));
	return { query, count: resultCount, country: normalizedCountry, freshness, results };
}

async function fetchPageContent(urlText: string): Promise<{ text: string; details: PageContentDetails }> {
	const response = await fetch(urlText, {
		headers: {
			"user-agent":
				"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
			accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
			"accept-language": "en-US,en;q=0.9",
		},
		signal: AbortSignal.timeout(15000),
	});
	if (!response.ok) throw new Error(`Page fetch failed: HTTP ${response.status}: ${response.statusText}`);

	const html = await response.text();
	const dom = new JSDOM(html, { url: urlText });
	const article = new Readability(dom.window.document).parse();

	if (article?.content) {
		const title = article.title ? `# ${article.title}\n\n` : "";
		const text = `${title}${htmlToMarkdown(article.content)}`.trim();
		return { text, details: { url: urlText, chars: text.length } };
	}

	const fallbackDoc = new JSDOM(html, { url: urlText });
	const body = fallbackDoc.window.document;
	body.querySelectorAll("script, style, noscript, nav, header, footer, aside").forEach((element) => {
		element.remove();
	});
	const title = body.querySelector("title")?.textContent?.trim();
	const main = body.querySelector("main, article, [role='main'], .content, #content") ?? body.body;
	const fallbackHtml = main?.innerHTML ?? "";
	if (fallbackHtml.trim().length <= 100) throw new Error("Could not extract readable content from this page.");
	const text = `${title ? `# ${title}\n\n` : ""}${htmlToMarkdown(fallbackHtml)}`.trim();
	return { text, details: { url: urlText, chars: text.length } };
}

export const webSearchTool: AgentTool<typeof webSearchParameters, WebSearchResultDetails> = {
	name: "web_search",
	label: "Web Search",
	description:
		"Search the web for current information using Brave Search. Use this when you need facts beyond memory.",
	executionMode: "sequential",
	parameters: webSearchParameters,
	execute: async (_id, params) => {
		const details = await searchWebWithBrave(
			params.query,
			params.count ?? 5,
			params.country ?? "DE",
			params.freshness,
		);
		return { content: [{ type: "text", text: formatSearchResults(details) }], details };
	},
};

export const pageContentTool: AgentTool<typeof pageContentParameters, PageContentDetails> = {
	name: "fetch_page_content",
	label: "Fetch Page Content",
	description: "Fetch readable markdown content from a specific URL found through web_search or provided by the user.",
	executionMode: "sequential",
	parameters: pageContentParameters,
	execute: async (_id, params) => {
		const result = await fetchPageContent(params.url);
		return {
			content: [{ type: "text", text: result.text || "No readable page content found." }],
			details: result.details,
		};
	},
};
