import { execFileSync } from "node:child_process";

const staged = execFileSync("git", ["diff", "--cached", "--name-only"], { encoding: "utf8" })
	.split("\n")
	.filter(Boolean);

const lockfiles = staged.filter((file) => file === "package-lock.json" || file.endsWith("/package-lock.json"));
if (lockfiles.length > 0 && process.env.PI_ALLOW_LOCKFILE_CHANGE !== "1") {
	console.error("Lockfile changes are staged:");
	for (const file of lockfiles) console.error(`  ${file}`);
	console.error("Set PI_ALLOW_LOCKFILE_CHANGE=1 if this lockfile change is intentional.");
	process.exit(1);
}
