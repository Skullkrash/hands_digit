const API_URL = "http://localhost:8000/predict";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");

const ctx = overlay.getContext("2d");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

const fpsElement = document.getElementById("fps");

let fps = 0;
let frameCount = 0;
let lastFpsUpdate = performance.now();

let processing = false;

async function init() {

    const stream =
        await navigator.mediaDevices.getUserMedia({
            video: {
                frameRate: { ideal: 40, max: 60 },
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });

    video.srcObject = stream;

    await video.play();

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    captureCanvas.width = 640;
    captureCanvas.height = 480;

    trackVideoFps();

    requestAnimationFrame(loop);
}

function trackVideoFps() {

    const onFrame = () => {
        frameCount++;

        const now = performance.now();

        if (now - lastFpsUpdate >= 100) {
            fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
            fpsElement.textContent = `${fps} FPS`;

            frameCount = 0;
            lastFpsUpdate = now;
        }

        if (typeof video.requestVideoFrameCallback === "function") {
            video.requestVideoFrameCallback(onFrame);
        }
    };

    if (typeof video.requestVideoFrameCallback === "function") {
        video.requestVideoFrameCallback(onFrame);
        return;
    }

    const fallback = () => {
        frameCount++;

        const now = performance.now();

        if (now - lastFpsUpdate >= 100) {
            fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
            fpsElement.textContent = `${fps} FPS`;

            frameCount = 0;
            lastFpsUpdate = now;
        }

        requestAnimationFrame(fallback);
    };

    requestAnimationFrame(fallback);
}

async function loop() {

    if (!processing) {

        processing = true;

        try {
            await sendFrame();
        }
        catch(error) {
            console.error(error);
        }
        finally {
            processing = false;
        }
    }

    requestAnimationFrame(loop);
}

async function sendFrame() {

    captureCtx.drawImage(
        video,
        0,
        0,
        captureCanvas.width,
        captureCanvas.height
    );

    const blob = await new Promise(resolve =>
        captureCanvas.toBlob(
            resolve,
            "image/jpeg",
            0.7
        )
    );

    const formData = new FormData();

    formData.append(
        "file",
        blob,
        "frame.jpg"
    );

    const response = await fetch(API_URL, {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    draw(data.detections);
}

function draw(detections) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const scaleX = overlay.width / captureCanvas.width;
    const scaleY = overlay.height / captureCanvas.height;

    let totalFingers = 0;

    for (const det of detections) {
        const x = det.x * scaleX;
        const y = det.y * scaleY;
        const w = det.w * scaleX;
        const h = det.h * scaleY;

        ctx.fillStyle = "rgba(99, 102, 241, 0.1)";
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 6);
        ctx.fill();

        ctx.strokeStyle = "#818cf8";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 6);
        ctx.stroke();

        const label = `${det.label}  ${Math.round(det.confidence * 100)}%`;
        ctx.font = "500 12px Inter, system-ui, sans-serif";
        const tw = ctx.measureText(label).width;
        const px = 8, py = 4, lh = 18;
        const lx = x;
        const ly = y > lh + py + 6 ? y - lh - py : y + h + py;

        ctx.fillStyle = "#6366f1";
        ctx.beginPath();
        ctx.roundRect(lx, ly, tw + px * 2, lh + py, 4);
        ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, lx + px, ly + lh - 2);

        const match = det.label[0];
        if (match) {
            totalFingers += parseInt(match, 10);
        }
    }

    document.getElementById("finger-count").textContent = totalFingers;
}

init();