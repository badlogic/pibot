import { readFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { extname, relative, resolve } from "node:path";

export interface HttpServer {
	server: Server;
}

function contentTypeFor(file: string): string {
	const extension = extname(file);
	if (extension === ".js") return "text/javascript; charset=utf-8";
	if (extension === ".css") return "text/css; charset=utf-8";
	if (extension === ".mp3") return "audio/mpeg";
	if (extension === ".wav") return "audio/wav";
	if (extension === ".webm") return "audio/webm";
	return "text/html; charset=utf-8";
}

async function serveStaticFile(
	req: IncomingMessage,
	res: ServerResponse,
	publicDir: string,
	version: string,
): Promise<void> {
	const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
	const path = url.pathname === "/" ? "/index.html" : url.pathname;
	const publicRoot = resolve(publicDir);
	const file = resolve(publicRoot, `.${path}`);
	const relativePath = relative(publicRoot, file);
	if (relativePath.startsWith("..") || relativePath.startsWith("/") || relativePath === "") {
		res.writeHead(403).end();
		return;
	}
	try {
		const data = await readFile(file);
		const extension = extname(file);
		res.writeHead(200, {
			"content-type": contentTypeFor(file),
			"cache-control": "no-store, no-cache, must-revalidate, proxy-revalidate",
			pragma: "no-cache",
			expires: "0",
		});
		if (extension === ".html") {
			res.end(
				data
					.toString("utf8")
					.replaceAll("style.css?v=dev", `style.css?v=${version}`)
					.replaceAll("app.js?v=dev", `app.js?v=${version}`),
			);
			return;
		}
		res.end(data);
	} catch {
		res.writeHead(404).end("not found");
	}
}

export function createHttpServer(deps: { publicDir: string; version: string }): HttpServer {
	const server = createServer(async (req, res) => {
		const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
		if (url.pathname === "/__version" && req.method === "GET") {
			res.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
			res.end(JSON.stringify({ version: deps.version }));
			return;
		}
		await serveStaticFile(req, res, deps.publicDir, deps.version);
	});
	return { server };
}
