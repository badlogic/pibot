export type LogSink = (entry: LogEntry) => void | Promise<void>;

export interface LogEntry {
	sequence: number;
	time: number;
	tags: string[];
	message: string;
	formatted: string;
}

interface LoggerState {
	sink: LogSink;
	sequence: number;
	queue: Promise<void>;
}

const reset = "\x1b[0m";
const colors = ["\x1b[36m", "\x1b[35m", "\x1b[33m", "\x1b[32m", "\x1b[34m", "\x1b[31m", "\x1b[90m"];

function tagColor(tag: string): string {
	let hash = 0;
	for (let i = 0; i < tag.length; i++) hash = (hash * 31 + tag.charCodeAt(i)) | 0;
	return colors[Math.abs(hash) % colors.length]!;
}

function formatTags(tags: string[], color: boolean): string {
	return tags.map((tag) => (color ? `${tagColor(tag)}[${tag}]${reset}` : `[${tag}]`)).join("");
}

function defaultSink(entry: LogEntry): void {
	console.log(formatEntry(entry, true));
}

export function formatEntry(entry: LogEntry, color = false): string {
	const prefix = formatTags(entry.tags, color);
	return prefix ? `${prefix} ${entry.message}` : entry.message;
}

export class Logger {
	constructor(
		private readonly state: LoggerState = { sink: defaultSink, sequence: 0, queue: Promise.resolve() },
		private readonly tags: string[] = [],
	) {}

	tag(tag: string): Logger {
		return new Logger(this.state, [...this.tags, tag]);
	}

	log(message: string): void {
		const entry = this.createEntry(message);
		this.state.queue = this.state.queue
			.catch(() => undefined)
			.then(() => this.state.sink(entry))
			.catch((error) =>
				console.error(`[logger] sink failed: ${error instanceof Error ? error.message : String(error)}`),
			);
	}

	async flush(): Promise<void> {
		await this.state.queue;
	}

	private createEntry(message: string): LogEntry {
		return {
			sequence: ++this.state.sequence,
			time: Date.now(),
			tags: this.tags,
			message,
			formatted: formatEntry({ sequence: 0, time: 0, tags: this.tags, message, formatted: "" }),
		};
	}
}

export function createLogger(sink?: LogSink): Logger {
	return new Logger(sink ? { sink, sequence: 0, queue: Promise.resolve() } : undefined);
}
