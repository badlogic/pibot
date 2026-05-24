import { watch } from "node:fs";
import { spawn } from "node:child_process";
import { resolve } from "node:path";
import * as esbuild from "esbuild";

const root = process.cwd();
const watchedDirs = [resolve(root, "src"), resolve(root, "public")];
const ignoredFiles = new Set([resolve(root, "public/app.js")]);
let server;
let restartTimer;
let restarting = false;
let pendingRestart = false;

function log(message) {
	console.log(`[dev] ${message}`);
}

async function compileClient() {
	await esbuild.build({
		entryPoints: [resolve(root, "src/client.ts")],
		bundle: true,
		format: "esm",
		target: "es2022",
		outfile: resolve(root, "public/app.js"),
	});
	log("client compiled");
}

function startServer() {
	server = spawn("node", ["--import", "tsx", resolve(root, "src/server.ts")], {
		cwd: root,
		env: process.env,
		stdio: "inherit",
	});
	server.on("exit", (code, signal) => {
		if (server) log(`server exited code=${code ?? "null"} signal=${signal ?? "null"}`);
	});
	log("server started");
}

async function stopServer() {
	if (!server) return;
	const child = server;
	server = undefined;
	await new Promise((resolveStop) => {
		const timeout = setTimeout(() => {
			child.kill("SIGKILL");
		}, 2000);
		child.once("exit", () => {
			clearTimeout(timeout);
			resolveStop();
		});
		child.kill("SIGTERM");
	});
}

async function restart(reason) {
	if (restarting) {
		pendingRestart = true;
		return;
	}
	restarting = true;
	try {
		log(`restart: ${reason}`);
		await stopServer();
		await compileClient();
		startServer();
	} catch (error) {
		console.error(error);
	} finally {
		restarting = false;
		if (pendingRestart) {
			pendingRestart = false;
			scheduleRestart("pending changes");
		}
	}
}

function scheduleRestart(reason) {
	if (restartTimer) clearTimeout(restartTimer);
	restartTimer = setTimeout(() => {
		restartTimer = undefined;
		restart(reason);
	}, 100);
}

function shouldIgnore(path) {
	return ignoredFiles.has(resolve(root, path));
}

for (const dir of watchedDirs) {
	watch(dir, { recursive: true }, (_eventType, filename) => {
		if (!filename) return;
		const path = resolve(dir, filename.toString());
		if (shouldIgnore(path)) return;
		scheduleRestart(path);
	});
	log(`watching ${dir}`);
}

process.on("SIGINT", async () => {
	await stopServer();
	process.exit(130);
});
process.on("SIGTERM", async () => {
	await stopServer();
	process.exit(143);
});

await restart("initial start");
