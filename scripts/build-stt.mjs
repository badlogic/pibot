import { spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const root = process.cwd();
const buildDir = resolve(root, "native/pibot-stt/build");
mkdirSync(buildDir, { recursive: true });

function run(command, args) {
	const result = spawnSync(command, args, { cwd: root, stdio: "inherit" });
	if (result.status !== 0) process.exit(result.status ?? 1);
}

run("cmake", ["-S", "native/pibot-stt", "-B", buildDir, "-DCMAKE_BUILD_TYPE=Release"]);
run("cmake", ["--build", buildDir, "--config", "Release", "-j"]);
