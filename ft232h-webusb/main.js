// Chrome/Edge WebUSB FT232H async bitbang demo.
// FT232H D4/D5 drive an H-bridge input pair.

const FTDI_VENDOR = 0x0403;
const FT232H_PRODUCT = 0x6014;

const SIO_RESET = 0x00;
const SIO_SET_BITMODE = 0x0b;
const BITMODE_RESET = 0x00;
const BITMODE_BITBANG = 0x01;
const INTERFACE_A = 1; // FTDI control-transfer wIndex for interface A

const D4 = 1 << 4;
const D5 = 1 << 5;
const DIRECTION_MASK = D4 | D5;

let device = null;
let outEndpoint = 0x02;
let altTimer = null;
let altState = false;

const logEl = document.querySelector("#log");
function log(msg) {
	logEl.textContent += `${new Date().toLocaleTimeString()} ${msg}\n`;
	logEl.scrollTop = logEl.scrollHeight;
}

async function ftdiControl(request, value, index = INTERFACE_A) {
	const result = await device.controlTransferOut({
		requestType: "vendor",
		recipient: "device",
		request,
		value,
		index,
	});
	if (result.status !== "ok") throw new Error(`controlTransferOut failed: ${result.status}`);
}

async function connect() {
	if (!("usb" in navigator)) {
		throw new Error(
			"WebUSB is not available. Use Chrome/Edge over http://localhost or https. Safari/Firefox will not work.",
		);
	}

	device = await navigator.usb.requestDevice({
		filters: [{ vendorId: FTDI_VENDOR, productId: FT232H_PRODUCT }],
	});

	await device.open();
	if (device.configuration === null) await device.selectConfiguration(1);

	// Claim interface 0 if possible. Some platforms/configs may not require it for vendor control writes,
	// but bulk writes generally do.
	try {
		await device.claimInterface(0);
	} catch (e) {
		log(`claimInterface warning: ${e.message}`);
	}

	const iface = device.configuration.interfaces[0];
	const alt = iface.alternate;
	const ep = alt.endpoints.find((e) => e.direction === "out" && e.type === "bulk");
	if (ep) outEndpoint = ep.endpointNumber;

	await ftdiControl(SIO_RESET, 0);
	await setBitbang(true);
	await writePins(0);
	log(`connected: ${device.productName || "FT232H"} endpoint OUT ${outEndpoint}`);
}

async function setBitbang(enabled) {
	const mode = enabled ? BITMODE_BITBANG : BITMODE_RESET;
	const value = DIRECTION_MASK | (mode << 8);
	await ftdiControl(SIO_SET_BITMODE, value);
}

async function writePins(value) {
	if (!device) throw new Error("Not connected");
	const data = new Uint8Array([value]);
	const result = await device.transferOut(outEndpoint, data);
	if (result.status !== "ok") throw new Error(`transferOut failed: ${result.status}`);
	log(`D4=${value & D4 ? 1 : 0} D5=${value & D5 ? 1 : 0}`);
}

async function stopAll() {
	if (altTimer) {
		clearInterval(altTimer);
		altTimer = null;
	}
	if (device) await writePins(0);
}

function wire(id, fn) {
	document.querySelector(id).addEventListener("click", async () => {
		try {
			await fn();
		} catch (e) {
			log(`ERROR: ${e.message}`);
			console.error(e);
		}
	});
}

wire("#connect", connect);
wire("#stop", stopAll);
wire("#fwd", () => writePins(D4));
wire("#rev", () => writePins(D5));
wire("#brake", () => writePins(D4 | D5));
wire("#altStop", stopAll);
wire("#alt", async () => {
	if (!device) throw new Error("Not connected");
	if (altTimer) clearInterval(altTimer);
	altState = false;
	await writePins(D4);
	altTimer = setInterval(async () => {
		try {
			altState = !altState;
			await writePins(altState ? D5 : D4);
		} catch (e) {
			log(`ERROR: ${e.message}`);
			clearInterval(altTimer);
			altTimer = null;
		}
	}, 5000);
});

window.addEventListener("beforeunload", () => {
	// Browser may not wait for this; use STOP before closing when motors are attached.
	if (device) device.transferOut(outEndpoint, new Uint8Array([0])).catch(() => {});
});

log("Load this page from http://localhost, click Connect FT232H, then use buttons.");
