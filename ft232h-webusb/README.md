# FT232H WebUSB H-Bridge Control

Experimental Chrome/Edge WebUSB controller for an Adafruit FT232H.

## Browser support

Use Chrome or Edge. Safari/Firefox do not support WebUSB. The page must be served from `http://localhost` or HTTPS, not opened as a `file://` URL.

Android Chrome may work with USB OTG, but permission/driver behavior varies by phone.

## Wiring

```text
FT232H D4  -> H-bridge IN1
FT232H D5  -> H-bridge IN2
FT232H GND -> H-bridge logic GND
```

Do not power the motor from the FT232H. Confirm the H-bridge accepts 3.3V logic HIGH.

## Run

```bash
cd webapp
chmod +x serve.sh
./serve.sh
```

Open:

```text
http://localhost:8000
```

Click **Connect FT232H**, choose the FT232H, then use the buttons.

## Emergency stop

The web app has a STOP button. From a terminal you can also force both pins low with the C helper:

```bash
cd ../macos
./ft232h_gpio stop 0
```
