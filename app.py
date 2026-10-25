
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

from src.detection import detect_objects
from src.prediction import train_demo_model, predict_waste
from src.explainability import explain_prediction
from src.routing import genetic_algorithm_route, compare_routes
from src.severity import calculate_severity
from src.evaluation import evaluate_regression


# ==================================================
# PROJECT FOLDERS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DETECTIONS_DIR = OUTPUT_DIR / "detections"
PLOTS_DIR = OUTPUT_DIR / "plots"
ROUTES_DIR = OUTPUT_DIR / "routes"

WASTE_FILE = DATA_DIR / "waste_records.csv"
LOCATIONS_FILE = DATA_DIR / "collection_locations.csv"

for folder in [
    DATA_DIR,
    OUTPUT_DIR,
    DETECTIONS_DIR,
    PLOTS_DIR,
    ROUTES_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ==================================================
# CSV FUNCTIONS
# ==================================================

def load_csv(file_path, columns):
    if not file_path.exists():
        pd.DataFrame(columns=columns).to_csv(file_path, index=False)

    try:
        df = pd.read_csv(file_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        df = pd.DataFrame(columns=columns)

    for column in columns:
        if column not in df.columns:
            df[column] = None

    return df


def load_waste_records():
    return load_csv(
        WASTE_FILE,
        ["Date", "Location", "Waste_kg"],
    )


def load_collection_locations():
    return load_csv(
        LOCATIONS_FILE,
        ["Location", "Latitude", "Longitude"],
    )


def save_csv(df, file_path):
    df.to_csv(file_path, index=False)


# ==================================================
# SEVERITY DISPLAY
# ==================================================

def display_severity_result(severity):
    st.subheader("Waste Severity Priority")

    a, b, c = st.columns(3)

    a.metric(
        "Detected Objects",
        severity["object_count"],
    )

    b.metric(
        "Severity Score",
        f'{severity["score"]}/100',
    )

    c.metric(
        "Waste Quantity",
        f'{severity["waste_kg"]:.2f} kg',
    )

    st.write(
        f'**Image Detection Score:** {severity["image_score"]}/50'
    )

    st.write(
        f'**Waste Quantity Score:** {severity["quantity_score"]}/50'
    )

    priority = severity["priority"]
    severity_name = severity["severity"]

    if severity_name == "High":
        st.error(
            f"Priority: {priority} — HIGH. "
            "Consider this location for earlier collection."
        )
    elif severity_name == "Medium":
        st.warning(
            f"Priority: {priority} — MEDIUM. "
            "Plan collection according to available resources."
        )
    elif severity_name == "Low":
        st.info(
            f"Priority: {priority} — LOW. "
            "Routine collection may be suitable."
        )
    else:
        st.success(
            "No Priority — no detected objects or recorded waste "
            "quantity contributed to this score."
        )

    st.caption(
        "Demonstration scoring only. These thresholds are not "
        "scientifically validated municipal waste standards."
    )


# ==================================================
# MAP FUNCTION
# ==================================================

def create_route_map(locations, route, optimized=True):
    route_points = []

    for index in route:
        row = locations.iloc[int(index)]

        route_points.append(
            (
                float(row["Latitude"]),
                float(row["Longitude"]),
                str(row["Location"]),
            )
        )

    if not route_points:
        return None

    center_lat = sum(point[0] for point in route_points) / len(route_points)
    center_lon = sum(point[1] for point in route_points) / len(route_points)

    route_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        control_scale=True,
    )

    for stop_number, (lat, lon, name) in enumerate(route_points, start=1):
        if stop_number == 1:
            marker_color = "green"
            stop_label = "Start / Depot"
        else:
            marker_color = "blue"
            stop_label = f"Stop {stop_number}"

        folium.Marker(
            location=[lat, lon],
            tooltip=f"{stop_label}: {name}",
            popup=folium.Popup(
                f"<b>{stop_label}</b><br>{name}<br>"
                f"Latitude: {lat:.6f}<br>"
                f"Longitude: {lon:.6f}",
                max_width=300,
            ),
            icon=folium.Icon(
                color=marker_color,
                icon="info-sign",
            ),
        ).add_to(route_map)

    line_points = [(point[0], point[1]) for point in route_points]

    if len(line_points) > 1:
        line_points.append(line_points[0])

        folium.PolyLine(
            locations=line_points,
            color="green" if optimized else "red",
            weight=5,
            opacity=0.85,
            tooltip=(
                "Optimized collection route"
                if optimized
                else "Original collection route"
            ),
        ).add_to(route_map)

    route_map.fit_bounds(
        [[point[0], point[1]] for point in route_points]
    )

    return route_map


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Waste Intel",
    page_icon="♻️",
    layout="wide",
)

st.title("♻️ Waste Intel")

st.subheader(
    "Explainable AI for Municipal Solid Waste Prediction "
    "and Dynamic Route Optimization"
)

st.caption(
    "YOLOv8 | XGBoost | SHAP | Genetic Algorithm | Route Mapping"
)

st.info(
    "This project uses demonstration models and straight-line GPS "
    "distance calculations. Map lines do not follow actual roads."
)


# ==================================================
# LOAD DATA AND MODEL
# ==================================================

waste_records = load_waste_records()
collection_locations = load_collection_locations()


@st.cache_resource
def get_prediction_model():
    return train_demo_model()


try:
    prediction_model = get_prediction_model()
except Exception as error:
    prediction_model = None
    st.error(f"Prediction model could not be loaded: {error}")


# ==================================================
# DASHBOARD METRICS
# ==================================================

total_waste = (
    pd.to_numeric(waste_records["Waste_kg"], errors="coerce")
    .fillna(0)
    .sum()
)

average_waste = (
    pd.to_numeric(waste_records["Waste_kg"], errors="coerce")
    .dropna()
    .mean()
)

if pd.isna(average_waste):
    average_waste = 0.0

m1, m2, m3, m4 = st.columns(4)

m1.metric("Waste Records", len(waste_records))
m2.metric("Total Waste", f"{total_waste:.2f} kg")
m3.metric("Average Waste", f"{average_waste:.2f} kg")
m4.metric("Collection Points", len(collection_locations))

st.divider()


# ==================================================
# TABS
# ==================================================

tabs = st.tabs(
    [
        "Waste Records",
        "YOLOv8 Detection",
        "Waste Prediction",
        "Collection Locations",
        "Route Optimization",
        "Model Evaluation",
        "Severity Priority",
    ]
)


# ==================================================
# TAB 1: WASTE RECORDS
# ==================================================

with tabs[0]:
    st.header("Waste Records")

    with st.form("waste_record_form"):
        record_date = st.date_input(
            "Collection Date",
            value=date.today(),
        )

        record_location = st.text_input("Location")

        record_waste = st.number_input(
            "Waste Collected (kg)",
            min_value=0.0,
            value=10.0,
            step=1.0,
        )

        save_record = st.form_submit_button("Save Waste Record")

    if save_record:
        if not record_location.strip():
            st.error("Please enter a location.")
        else:
            new_record = pd.DataFrame(
                [
                    {
                        "Date": str(record_date),
                        "Location": record_location.strip(),
                        "Waste_kg": record_waste,
                    }
                ]
            )

            updated = pd.concat(
                [load_waste_records(), new_record],
                ignore_index=True,
            )

            save_csv(updated, WASTE_FILE)
            st.success("Waste record saved!")
            st.rerun()

    waste_records = load_waste_records()

    st.subheader("Saved Waste Records")

    if not waste_records.empty:
        st.dataframe(waste_records, use_container_width=True)

        chart_df = waste_records.copy()
        chart_df["Waste_kg"] = pd.to_numeric(
            chart_df["Waste_kg"], errors="coerce"
        ).fillna(0)

        st.bar_chart(
            chart_df.groupby("Location")["Waste_kg"].sum()
        )

        st.download_button(
            "Download Waste Records",
            waste_records.to_csv(index=False).encode("utf-8"),
            file_name="waste_records.csv",
            mime="text/csv",
        )
    else:
        st.info("No waste records available.")


# ==================================================
# TAB 2: YOLOV8 DETECTION + SEVERITY
# ==================================================

with tabs[1]:
    st.header("YOLOv8 Object Detection")

    st.write(
        "Upload an image, detect objects, and combine the detection "
        "count with the recorded waste quantity to estimate priority."
    )

    uploaded_image = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        key="detection_image",
    )

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.10,
        step=0.05,
    )

    # The user can enter the measured quantity for this image/location.
    waste_kg = st.number_input(
        "Recorded Waste Quantity (kg)",
        min_value=0.0,
        value=10.0,
        step=1.0,
        key="detection_waste_kg",
    )

    if uploaded_image is not None:
        try:
            from PIL import Image

            original_image = Image.open(uploaded_image).convert("RGB")

            with st.spinner("Detecting objects..."):
                annotated_image, detections = detect_objects(
                    original_image,
                    confidence=confidence,
                )

            left, right = st.columns(2)

            with left:
                st.image(
                    original_image,
                    caption="Uploaded Image",
                    use_container_width=True,
                )

            with right:
                st.image(
                    annotated_image,
                    caption="Detection Results",
                    use_container_width=True,
                )

            severity = calculate_severity(
                detections,
                waste_kg=waste_kg,
            )

            display_severity_result(severity)

            if detections:
                detection_df = pd.DataFrame(detections)
                st.subheader("Detected Objects")
                st.dataframe(detection_df, use_container_width=True)

                st.download_button(
                    "Download Detection CSV",
                    detection_df.to_csv(index=False).encode("utf-8"),
                    file_name="detection_results.csv",
                    mime="text/csv",
                )
            else:
                st.info("No objects were detected in this image.")

            st.warning(
                "The default YOLOv8 model is a general object detector, "
                "not a specially trained waste-classification model. "
                "Object count alone does not measure actual waste mass."
            )

        except Exception as error:
            st.error(f"Detection failed: {error}")
    else:
        st.info(
            "Upload an image to calculate severity using detections "
            "and the waste quantity entered above."
        )


# ==================================================
# TAB 3: WASTE PREDICTION AND SHAP
# ==================================================

with tabs[2]:
    st.header("Waste Generation Prediction")

    col1, col2 = st.columns(2)

    with col1:
        households = st.number_input(
            "Number of Households",
            min_value=1,
            max_value=1000000,
            value=500,
            step=50,
        )

    with col2:
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda day: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ][day],
        )

    if st.button("Predict Waste"):
        if prediction_model is None:
            st.error("Prediction model is unavailable.")
        else:
            try:
                prediction = predict_waste(
                    prediction_model,
                    households,
                    day_of_week,
                )

                st.success(f"Predicted waste: {prediction:.2f} kg")

                baseline, explanation = explain_prediction(
                    prediction_model,
                    households,
                    day_of_week,
                )

                st.write(f"Model baseline: {baseline:.2f} kg")
                st.subheader("SHAP Explanation")
                st.dataframe(explanation, use_container_width=True)

                if (
                    "Feature" in explanation.columns
                    and "SHAP impact (kg)" in explanation.columns
                ):
                    st.bar_chart(
                        explanation.set_index("Feature")[
                            "SHAP impact (kg)"
                        ]
                    )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

    st.warning(
        "The prediction model uses synthetic demonstration data. "
        "It is not a validated municipal waste forecast."
    )


# ==================================================
# TAB 4: COLLECTION LOCATIONS
# ==================================================

with tabs[3]:
    st.header("Collection Locations")

    st.write(
        "Add collection points using their GPS coordinates. "
        "The first saved point is the starting depot."
    )

    with st.form("collection_location_form"):
        location_name = st.text_input("Location Name")

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=17.3850,
            format="%.6f",
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=78.4867,
            format="%.6f",
        )

        save_location = st.form_submit_button(
            "Save Collection Location"
        )

    if save_location:
        if not location_name.strip():
            st.error("Please enter a location name.")
        else:
            current = load_collection_locations()

            duplicate = (
                current["Location"]
                .fillna("")
                .astype(str)
                .str.casefold()
                .eq(location_name.strip().casefold())
            ).any()

            if duplicate:
                st.error("This location name already exists.")
            else:
                new_location = pd.DataFrame(
                    [
                        {
                            "Location": location_name.strip(),
                            "Latitude": latitude,
                            "Longitude": longitude,
                        }
                    ]
                )

                current = pd.concat(
                    [current, new_location],
                    ignore_index=True,
                )

                save_csv(current, LOCATIONS_FILE)
                st.success("Location saved!")
                st.rerun()

    collection_locations = load_collection_locations()

    st.subheader("Saved Collection Points")

    if not collection_locations.empty:
        st.dataframe(
            collection_locations,
            use_container_width=True,
        )

        st.download_button(
            "Download Locations CSV",
            collection_locations.to_csv(index=False).encode("utf-8"),
            file_name="collection_locations.csv",
            mime="text/csv",
        )
    else:
        st.info("No collection locations added yet.")


# ==================================================
# TAB 5: ROUTE OPTIMIZATION AND MAP
# ==================================================

with tabs[4]:
    st.header("Route Optimization and Tracking")

    st.write(
        "Optimize the visiting order, compare distances, and display "
        "the collection route on a map."
    )

    collection_locations = load_collection_locations()

    if len(collection_locations) < 2:
        st.warning(
            "Add at least two collection points in the "
            "'Collection Locations' tab first."
        )
    else:
        valid_locations = collection_locations.copy()

        valid_locations["Latitude"] = pd.to_numeric(
            valid_locations["Latitude"], errors="coerce"
        )
        valid_locations["Longitude"] = pd.to_numeric(
            valid_locations["Longitude"], errors="coerce"
        )

        valid_locations = valid_locations.dropna(
            subset=["Latitude", "Longitude"]
        ).reset_index(drop=True)

        valid_locations = valid_locations[
            valid_locations["Latitude"].between(-90, 90)
            & valid_locations["Longitude"].between(-180, 180)
        ].reset_index(drop=True)

        if len(valid_locations) < 2:
            st.error(
                "At least two locations need valid latitude and longitude."
            )
        else:
            st.subheader("Locations Used for Routing")
            st.dataframe(
                valid_locations[
                    ["Location", "Latitude", "Longitude"]
                ],
                use_container_width=True,
            )

            generation_count = st.slider(
                "Genetic Algorithm Generations",
                min_value=50,
                max_value=500,
                value=250,
                step=50,
            )

            population_count = st.slider(
                "Population Size",
                min_value=20,
                max_value=150,
                value=80,
                step=10,
            )

            if st.button("Optimize Collection Route"):
                try:
                    coordinates = list(
                        zip(
                            valid_locations["Latitude"].astype(float),
                            valid_locations["Longitude"].astype(float),
                        )
                    )

                    with st.spinner("Optimizing route..."):
                        best_route, best_distance = genetic_algorithm_route(
                            coordinates,
                            population_size=population_count,
                            generations=generation_count,
                            mutation_rate=0.15,
                            seed=42,
                        )

                        comparison = compare_routes(
                            coordinates,
                            best_route,
                        )

                    st.session_state["route_result"] = {
                        "route": list(best_route),
                        "distance": float(best_distance),
                        "comparison": comparison,
                        "locations": valid_locations.to_dict(
                            orient="records"
                        ),
                    }

                    st.success("Route optimization completed!")

                except Exception as error:
                    st.error(f"Route optimization failed: {error}")

            if "route_result" in st.session_state:
                result = st.session_state["route_result"]

                route = result["route"]
                comparison = result["comparison"]
                route_locations = pd.DataFrame(result["locations"])

                st.divider()
                st.subheader("Distance Comparison")

                a, b = st.columns(2)

                a.metric(
                    "Original Route",
                    f"{comparison['original_distance_km']:.2f} km",
                )

                b.metric(
                    "Optimized Route",
                    f"{comparison['optimized_distance_km']:.2f} km",
                )

                c, d = st.columns(2)

                c.metric(
                    "Distance Saved",
                    f"{comparison['distance_saved_km']:.2f} km",
                )

                d.metric(
                    "Improvement",
                    f"{comparison['improvement_percent']:.2f}%",
                )

                chart_df = pd.DataFrame(
                    {
                        "Route": ["Original", "Optimized"],
                        "Distance (km)": [
                            comparison["original_distance_km"],
                            comparison["optimized_distance_km"],
                        ],
                    }
                ).set_index("Route")

                st.bar_chart(chart_df)

                original_route = list(range(len(route_locations)))

                map_tab1, map_tab2 = st.tabs(
                    ["Optimized Route Map", "Original Route Map"]
                )

                with map_tab1:
                    optimized_map = create_route_map(
                        route_locations,
                        route,
                        optimized=True,
                    )

                    if optimized_map is not None:
                        st_folium(
                            optimized_map,
                            width=None,
                            height=550,
                            key="optimized_route_map",
                        )

                with map_tab2:
                    original_map = create_route_map(
                        route_locations,
                        original_route,
                        optimized=False,
                    )

                    if original_map is not None:
                        st_folium(
                            original_map,
                            width=None,
                            height=550,
                            key="original_route_map",
                        )

                st.subheader("Optimized Stop Order")

                route_rows = []

                for stop_number, index in enumerate(route, start=1):
                    row = route_locations.iloc[int(index)]

                    route_rows.append(
                        {
                            "Stop": stop_number,
                            "Location": row["Location"],
                            "Latitude": row["Latitude"],
                            "Longitude": row["Longitude"],
                        }
                    )

                route_df = pd.DataFrame(route_rows)

                st.dataframe(route_df, use_container_width=True)

                stop_names = route_df["Location"].astype(str).tolist()

                if stop_names:
                    st.markdown(
                        "**Route:** "
                        + " → ".join(stop_names + [stop_names[0]])
                    )

                st.download_button(
                    "Download Optimized Route CSV",
                    route_df.to_csv(index=False).encode("utf-8"),
                    file_name="optimized_route.csv",
                    mime="text/csv",
                )

                comparison_df = pd.DataFrame(
                    [
                        {
                            "Original distance (km)": (
                                comparison["original_distance_km"]
                            ),
                            "Optimized distance (km)": (
                                comparison["optimized_distance_km"]
                            ),
                            "Distance saved (km)": (
                                comparison["distance_saved_km"]
                            ),
                            "Improvement (%)": (
                                comparison["improvement_percent"]
                            ),
                        }
                    ]
                )

                st.download_button(
                    "Download Distance Comparison CSV",
                    comparison_df.to_csv(index=False).encode("utf-8"),
                    file_name="route_comparison.csv",
                    mime="text/csv",
                )

                st.info(
                    "Map lines connect GPS points directly. The map does "
                    "not show live vehicle tracking or road navigation."
                )


# ==================================================
# TAB 6: MODEL EVALUATION
# ==================================================

with tabs[5]:
    st.header("Model Evaluation")

    actual_values = [100, 150, 200, 250, 300]
    predicted_values = [110, 140, 190, 260, 290]

    metrics = evaluate_regression(
        actual_values,
        predicted_values,
    )

    a, b, c = st.columns(3)

    a.metric("MAE", f"{metrics['MAE']:.2f} kg")
    b.metric("RMSE", f"{metrics['RMSE']:.2f} kg")

    if metrics["R2"] is not None:
        c.metric("R² Score", f"{metrics['R2']:.4f}")

    evaluation_df = pd.DataFrame(
        {
            "Actual Waste (kg)": actual_values,
            "Predicted Waste (kg)": predicted_values,
        }
    )

    st.subheader("Actual vs Predicted")
    st.dataframe(evaluation_df, use_container_width=True)

    st.line_chart(
        evaluation_df.rename(
            columns={
                "Actual Waste (kg)": "Actual",
                "Predicted Waste (kg)": "Predicted",
            }
        )
    )

    st.download_button(
        "Download Evaluation CSV",
        evaluation_df.to_csv(index=False).encode("utf-8"),
        file_name="model_evaluation.csv",
        mime="text/csv",
    )

    st.warning(
        "These are illustrative example metrics, not measured accuracy "
        "from validation on real municipal waste data."
    )


# ==================================================
# TAB 7: SEVERITY PRIORITY
# ==================================================

with tabs[6]:
    st.header("Waste Severity Priority")

    st.write(
        "Calculate collection priority from a detection count and "
        "a measured waste quantity. This tab is useful when you already "
        "have a detection result or a recorded waste measurement."
    )

    manual_count = st.number_input(
        "Number of Detected Objects",
        min_value=0,
        max_value=100000,
        value=3,
        step=1,
        key="manual_detection_count",
    )

    manual_waste_kg = st.number_input(
        "Recorded Waste Quantity (kg)",
        min_value=0.0,
        value=75.0,
        step=1.0,
        key="manual_severity_waste",
    )

    if st.button("Calculate Collection Priority"):
        # Create a list of the requested length for the severity function.
        manual_detections = [
            {"name": "detected_object"}
            for _ in range(int(manual_count))
        ]

        try:
            manual_severity = calculate_severity(
                manual_detections,
                waste_kg=manual_waste_kg,
            )

            st.session_state["manual_severity_result"] = manual_severity

        except Exception as error:
            st.error(f"Severity calculation failed: {error}")

    if "manual_severity_result" in st.session_state:
        display_severity_result(
            st.session_state["manual_severity_result"]
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Waste Intel | YOLOv8 | XGBoost | SHAP | "
    "Genetic Algorithm | Severity Priority | Route Map"
)
