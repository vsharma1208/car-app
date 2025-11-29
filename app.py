import gradio as gr
import pandas as pd
from catboost import CatBoostRegressor

# -----------------------
# LOAD CATBOOST MODEL
# -----------------------
model = CatBoostRegressor()
model.load_model("car_price_model.cbm")

# Allowed dropdown values
MANUFACTURERS = sorted([
    "Toyota","Honda","Ford","BMW","Mercedes-Benz","Audi","Volkswagen","Hyundai",
    "Kia","Chevrolet","Nissan","Lexus","Mazda","Jeep","Porsche","Land Rover",
    "Volvo","Mini","Jaguar","Ferrari","Bentley","Subaru","Mitsubishi"
])

FUEL_TYPES = ["Petrol", "Diesel", "Hybrid", "Electric"]

# -----------------------
# PREDICTION FUNCTION
# -----------------------
def predict_price(image, manufacturer, model_name, year, fuel, mileage):

    # Build full row with ALL columns in training order
    row = {
        "ID": 0,                              # dummy id
        "Levy": 0,
        "Manufacturer": manufacturer,
        "Model": model_name,
        "Prod. year": int(year),
        "Category": "Unknown",
        "Leather interior": "Unknown",
        "Fuel type": fuel,
        "Engine volume": 0,
        "Mileage": float(mileage),
        "Cylinders": 0,
        "Gear box type": "Unknown",
        "Drive wheels": "Unknown",
        "Doors": "Unknown",
        "Wheel": "Unknown",
        "Color": "Unknown",
        "Airbags": 0,
    }

    # Force correct order (important!)
    column_order = [
        "ID", "Levy", "Manufacturer", "Model", "Prod. year",
        "Category", "Leather interior", "Fuel type", "Engine volume",
        "Mileage", "Cylinders", "Gear box type", "Drive wheels",
        "Doors", "Wheel", "Color", "Airbags"
    ]

    df = pd.DataFrame([row])[column_order]

    # Predict
    price = model.predict(df)[0]
    low = price * 0.9
    high = price * 1.1

    return f"""
### 💰 Estimated Price: **${price:,.0f}**
#### 📉 Confidence Range: ${low:,.0f} – ${high:,.0f}
"""

# -----------------------
# GRADIO UI — SAME LAYOUT
# -----------------------
with gr.Blocks() as demo:

    gr.Markdown("# Car Price Estimator")
    gr.Markdown("Upload a car photo and enter details to estimate its price.")

    with gr.Row():

        # LEFT — Image + Prediction
        with gr.Column(scale=1):
            image = gr.Image(label="Car Image", type="filepath", height=350)
            prediction_box = gr.Markdown("### Prediction will appear here...")

        # RIGHT — Inputs + Predict
        with gr.Column(scale=1):
            manufacturer = gr.Dropdown(MANUFACTURERS, label="Manufacturer")
            model_name = gr.Textbox(label="Model")
            year = gr.Slider(1990, 2025, value=2018, label="Year")
            fuel = gr.Dropdown(FUEL_TYPES, label="Fuel Type")
            mileage = gr.Number(label="Mileage (km)")
            btn = gr.Button("Predict Price")

    btn.click(
        predict_price,
        inputs=[image, manufacturer, model_name, year, fuel, mileage],
        outputs=prediction_box
    )

demo.launch()