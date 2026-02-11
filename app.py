import gradio as gr
import tempfile
from ultralytics import YOLO
from src.config import RAW_VIDEO, PROCESSED_VIDEO
from src.core.video_processor import VideoProcessor
from src.core.image_detector import ObjectDetection
from pathlib import Path


# Auto-download model on first run
MODEL_PATH = Path("models/yolov8m.pt")
MODEL_PATH.parent.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    print("📥 Downloading YOLOv8 model...")
    temp_model = YOLO("yolov8m.pt")

    # Model downloaded to cache
    import shutil
    shutil.copy(
        Path.home() / ".cache/ultralytics/weights/yolov8m.pt",
        MODEL_PATH
    )
    print("Model ready!")


# Model
detector = ObjectDetection(str(Path("models/yolov8m.pt")))
detector.model.verbose = False
processor = VideoProcessor(detector)

def process_user_video(video_path):
    """
    Processing user video file
    """
    if video_path is None:
        return None, "Please upload a video first"

    # Temprorary file for user upload
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    temp_output.close()

    # Process
    processor.process_video(video_path, output_path)

    return output_path, "✅ Processing complete!"

with gr.Blocks() as demo:
    # Header
    gr.HTML("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🚦 JustCV
            </h1>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 1.2em;">
                Urban Traffic Monitor powered by YOLOv8
            </p>
        </div>
    """)

    with gr.Tabs():
        # TAB 1: Demo
        with gr.Tab("📊 Live Demo"):
            gr.Markdown("### Side-by-side Comparison")
            gr.Markdown("Watch real-time object detection on urban traffic footage")

            with gr.Row():
                gr.Video(
                    value=str(RAW_VIDEO),
                    label="📹 Raw Footage",
                    autoplay=True,
                    loop=True
                )
                gr.Video(
                    value=str(PROCESSED_VIDEO),
                    label="🎯 Object Detection",
                    autoplay=True,
                    loop=True
                )

        # TAB 2: Upload & Process
        with gr.Tab("🎬 Process Your Video"):
            gr.Markdown("### Upload your own traffic video")
            gr.Markdown("Supported formats: MP4, AVI, MOV")

            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    process_btn = gr.Button("🚀 Process Video", variant="primary")
                    status_msg = gr.Textbox(label="Status", interactive=False)

                with gr.Column():
                    video_output = gr.Video(label="Processed Result")

            process_btn.click(
                process_user_video,
                inputs=video_input,
                outputs=[video_output, status_msg]
            )

        # TAB 3: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## 🚦 JustCV - Urban Traffic Monitor
            
            **Real-time AI-powered traffic analysis system**
            
            ### 🎯 Features
            
            - **Object Detection**: Detects vehicles, pedestrians, and bicycles using YOLOv8
            - **Real-time Processing**: Processes video streams frame-by-frame
            - **Multi-class Recognition**: Identifies cars, buses, trucks, motorcycles, bicycles, and people
            - **Side-by-side View**: Compare raw footage with detection results
            - **Custom Video Upload**: Process your own traffic videos
            
            ### 🛠️ Tech Stack
            
            - **Python 3.13** - Core programming language
            - **YOLOv8 (Medium)** - State-of-the-art object detection model
            - **OpenCV** - Video processing and computer vision
            - **PyTorch + CUDA** - GPU-accelerated deep learning
            - **Gradio** - Web interface framework
            
            ### 📊 Detected Classes
            
            The system detects and counts the following objects:
            - 🚶 **Pedestrians** - People walking or standing
            - 🚗 **Cars** - Personal vehicles
            - 🚌 **Buses** - Public transport
            - 🚚 **Trucks** - Commercial vehicles
            - 🏍️ **Motorcycles** - Two-wheeled motorized vehicles
            - 🚴 **Bicycles** - Non-motorized two-wheelers
            
            *(Traffic lights are excluded from detection)*
            
            ### 🎓 Project Background
            
            JustCV was built as a portfolio project to demonstrate:
            - Computer vision pipeline development
            - Real-time video processing
            - Deep learning model deployment
            - Production-ready code architecture
            
            ### 🔗 Links
            
            - **GitHub**: [github.com/justgodtears/JustCV](https://github.com/justgodtears/JustCV)
            - **LinkedIn**: [Oliwier Opyrchal](https://linkedin.com/in/oliwier-opyrchal)
            
            ### 📝 Citation
            
            If you use YOLOv8 model:
            ```
            @software{yolov8_ultralytics,
              author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
              title = {Ultralytics YOLOv8},
              year = {2023},
              url = {https://github.com/ultralytics/ultralytics}
            }
            ```
            
            ---
            
            Built with ❤️ for urban traffic analysis
            """)

demo.launch(theme=gr.themes.Soft())