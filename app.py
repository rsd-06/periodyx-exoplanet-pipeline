import gradio as gr
import os
import matplotlib.pyplot as plt
from PIL import Image

from data.synthetic import make_synthetic_lightcurve, make_false_positive_lightcurve
from pipeline import run_pipeline
from visualization.plots import plot_detection_summary
from characterization.trapezoid_fit import _trapezoid

os.makedirs("demo_outputs", exist_ok=True)

def run_exoplanet_pipeline(test_case):
    if test_case == "Synthetic Transit (Planet)":
        t, flux, truth = make_synthetic_lightcurve()
        target_name = "SYNTH-TRANSIT-01"
        save_path = "demo_outputs/case1_transit.png"
    else:
        t, flux, truth = make_false_positive_lightcurve()
        target_name = "SYNTH-FALSEPOS-01"
        save_path = "demo_outputs/case2_falsepositive.png"

    # Run the full pipeline
    result = run_pipeline(t, flux, target_name=target_name, use_tls=True)

    # Plot the result
    arrs = result["_arrays"]
    plot_detection_summary(
        arrs["time"], arrs["flux_flat"], arrs["phase_time"], arrs["phase_flux"],
        _trapezoid, arrs["fit_params"], target_name=target_name,
        save_path=save_path
    )
    
    img = Image.open(save_path)
    
    # Format text output
    output_text = f"## Classification Result: **{result['classification']}**\n\n"
    output_text += "### Pipeline Diagnostics\n"
    output_text += f"- **BLS Best Period:** {result['bls']['best_period']:.4f} days\n"
    if "tls" in result:
        output_text += f"- **TLS SDE (Significance):** {result['tls']['SDE']:.2f}\n"
    
    fit = result["trapezoid_fit"]
    output_text += "\n### Physical Characterization (Trapezoid Fit)\n"
    output_text += f"- **Depth:** {fit['depth']*100:.4f}%\n"
    output_text += f"- **Total Duration:** {fit['t_tot']*24:.3f} hours\n"
    output_text += f"- **Ingress Fraction:** {fit['ingress_fraction']:.3f}\n"

    feats = result["features"]
    output_text += "\n### Discriminator Features\n"
    output_text += f"- **Odd-Even Depth Difference:** {feats['odd_even_depth_diff']:.4f}\n"
    output_text += f"- **Secondary Eclipse Depth:** {feats['secondary_eclipse_depth']:.6f}\n"

    return img, output_text

with gr.Blocks(title="PeriodyX Exoplanet Pipeline") as demo:
    gr.Markdown("# 🪐 PeriodyX Exoplanet Detection Pipeline")
    gr.Markdown("This interactive demo runs the full end-to-end signal processing and physical characterization pipeline on synthetic light curves. Select a test case below to see how the pipeline detrends, searches, and classifies the signal.")
    
    with gr.Row():
        test_case = gr.Radio(
            ["Synthetic Transit (Planet)", "Synthetic False Positive (Eclipsing Binary)"],
            label="Select a Light Curve to Process",
            value="Synthetic Transit (Planet)"
        )
        btn = gr.Button("Run Pipeline", variant="primary")
        
    with gr.Row():
        with gr.Column(scale=2):
            output_image = gr.Image(label="Pipeline Output Visualization")
        with gr.Column(scale=1):
            output_markdown = gr.Markdown(label="Results & Extracted Features")
            
    btn.click(fn=run_exoplanet_pipeline, inputs=test_case, outputs=[output_image, output_markdown])

if __name__ == "__main__":
    demo.launch()
