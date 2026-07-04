const CameraPipeline = {
    camera: null,
    hands: null,
    canvas: null,
    ctx: null,
    video: null,
    isPredicting: false,

    onResultCallback: null,

    async start(videoEl, canvasEl, onResult, onError) {
        this.video = videoEl;
        this.canvas = canvasEl;
        this.ctx = canvasEl.getContext('2d');
        this.onResultCallback = onResult;

        try {
            // 1. Initialize MediaPipe Hands
            this.hands = new Hands({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
                }
            });

            this.hands.setOptions({
                maxNumHands: 2,
                modelComplexity: 1,
                minDetectionConfidence: 0.7,
                minTrackingConfidence: 0.7
            });

            this.hands.onResults((results) => this.onMediaPipeResults(results));

            // 2. Initialize Camera Utils
            this.camera = new Camera(this.video, {
                onFrame: async () => {
                    await this.hands.send({ image: this.video });
                },
                width: 640,
                height: 480,
                facingMode: "user"
            });

            await this.camera.start();

            // Align canvas resolution with video stream
            const vw = this.video.videoWidth || 640;
            const vh = this.video.videoHeight || 480;
            this.canvas.width = vw;
            this.canvas.height = vh;

        } catch (err) {
            console.error("Error starting Camera/MediaPipe:", err);
            if (onError) onError(err);
        }
    },

    stop() {
        if (this.camera) {
            this.camera.stop();
            this.camera = null;
        }
        if (this.hands) {
            this.hands.close();
            this.hands = null;
        }
        if (this.ctx && this.canvas) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    },

    async onMediaPipeResults(results) {
        // Setup canvas for drawing
        this.ctx.save();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // We do NOT draw the video image on the canvas.
        // The <video> element underneath handles the video rendering, and CSS layers them.

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            for (const landmarks of results.multiHandLandmarks) {
                // Add a slight glow effect to the context
                this.ctx.shadowColor = '#10b981';
                this.ctx.shadowBlur = 10;

                // Draw connections (skeleton)
                drawConnectors(this.ctx, landmarks, HAND_CONNECTIONS, {
                    color: '#10b981', // Accent green
                    lineWidth: 4
                });

                // Draw landmarks (points)
                this.ctx.shadowColor = '#ef4444';
                this.ctx.shadowBlur = 8;
                drawLandmarks(this.ctx, landmarks, {
                    color: '#ef4444', // Accent red
                    lineWidth: 2,
                    fillColor: '#ffffff',
                    radius: 4
                });
            }

            // Send coords to backend if not throttling
            if (!this.isPredicting) {
                this.isPredicting = true;
                this.sendToBackend(results.multiHandLandmarks);
            }
        }
        this.ctx.restore();
    },

    async sendToBackend(hands) {
        // Use a relative URL for local development/deployment consistency
        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                landmarks: hands
            })
        })
        .then(res => res.json())
        .then(data => {
            console.log(data);
            
            // Update UI elements as requested
            const predEl = document.getElementById("prediction-letter");
            const confEl = document.getElementById("confidence-pct");
            
            if (predEl) predEl.innerText = data.prediction || "—";
            if (confEl) confEl.innerText = `${(data.confidence * 100).toFixed(1)}%`;
            
            // Note: In a full implementation, we'd also update the confidence bar and top-3 suggestions
            if (this.onResultCallback) {
                this.onResultCallback(data);
            }
        })
        .catch(err => console.error("Backend prediction error:", err))
        .finally(() => {
            // Limit request rate to ~4 Hz (250ms cooldown)
            setTimeout(() => { this.isPredicting = false; }, 250);
        });
    }
};

window.CameraPipeline = CameraPipeline;
