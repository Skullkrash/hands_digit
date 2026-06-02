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
    ctx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    const scaleX =
        overlay.width /
        captureCanvas.width;

    const scaleY =
        overlay.height /
        captureCanvas.height;

    for (const det of detections) {

        const x = det.x * scaleX;
        const y = det.y * scaleY;
        const w = det.w * scaleX;
        const h = det.h * scaleY;

        ctx.strokeStyle = "lime";
        ctx.lineWidth = 3;

        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "lime";
        ctx.font = "18px Arial";

        ctx.fillText(
            `${det.label} (${det.confidence})`,
            x,
            y - 10
        );
    }
}

init();