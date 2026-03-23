
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load models
maintenance_model = joblib.load("maintenance_model.pkl")
failure_model = joblib.load("failure_model.pkl")



# Page config
st.set_page_config(page_title="Vehicle Health Monitor", layout="wide")

st.title(" AI Vehicle Health Monitoring System")

st.markdown("### Smart Prediction of Maintenance & Machine Failure")

st.divider()

# Create two columns
col1, col2 = st.columns(2)

# -------- LEFT COLUMN (Vehicle Data) --------
with col1:
    st.subheader(" Vehicle Details")

    vehicle_model = st.number_input("Vehicle Model", 0, 10, 1)
    mileage = st.number_input("Mileage", 0, 200000, 50000)
    vehicle_age = st.number_input("Vehicle Age", 0, 20, 5)

    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
    fuel_map = {"Petrol":0, "Diesel":1, "Electric":2}
    fuel_type = fuel_map[fuel_type]

    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    trans_map = {"Manual":0, "Automatic":1}
    transmission = trans_map[transmission]

    engine_size = st.number_input("Engine Size", 0.5, 5.0, 2.0)
    odometer = st.number_input("Odometer Reading", 0, 200000, 60000)
    owner_type = st.number_input("Owner Type", 0, 2, 1)

    insurance = st.number_input("Insurance Premium", 0, 50000, 15000)
    accident = st.selectbox("Accident History", ["No", "Yes"])
    accident = 1 if accident == "Yes" else 0

    fuel_eff = st.number_input("Fuel Efficiency", 5, 30, 18)

    tire = st.selectbox("Tire Condition", ["Good", "Average", "Bad"])
    tire_map = {"Good":2, "Average":1, "Bad":0}
    tire = tire_map[tire]

    brake = st.selectbox("Brake Condition", ["Good", "Average", "Bad"])
    brake_map = {"Good":2, "Average":1, "Bad":0}
    brake = brake_map[brake]

    battery = st.selectbox("Battery Status", ["Good", "Average", "Weak"])
    battery_map = {"Good":2, "Average":1, "Weak":0}
    battery = battery_map[battery]

    last_service = st.number_input("Last Service Year", 2000, 2025, 2023)
    warranty = st.number_input("Warranty Expiry Year", 2000, 2030, 2026)

# -------- RIGHT COLUMN (Sensor Data) --------
with col2:
    st.subheader(" Machine Sensor Data")

    machine_type = st.selectbox("Machine Type", ["Low", "Medium", "High"])
    type_map = {"Low":0, "Medium":1, "High":2}
    machine_type = type_map[machine_type]

    air_temp = st.number_input("Air Temperature (K)", 250.0, 350.0, 298.0)
    process_temp = st.number_input("Process Temperature (K)", 250.0, 350.0, 308.0)
    rpm = st.number_input("Rotational Speed (rpm)", 1000, 3000, 1500)
    torque = st.number_input("Torque (Nm)", 0.0, 100.0, 40.0)
    tool_wear = st.number_input("Tool Wear (min)", 0, 300, 20)

st.divider()

# -------- PREDICTION BUTTON --------
if st.button(" Predict Now"):
    vehicle_data = pd.DataFrame([[vehicle_model, mileage, vehicle_age,
        fuel_type, transmission, engine_size, odometer, owner_type,
        insurance, accident, fuel_eff, tire, brake, battery,
        last_service, warranty]],
        columns=[
        'Vehicle_Model','Mileage','Vehicle_Age','Fuel_Type',
        'Transmission_Type','Engine_Size','Odometer_Reading',
        'Owner_Type','Insurance_Premium','Accident_History',
        'Fuel_Efficiency','Tire_Condition','Brake_Condition',
        'Battery_Status','Last_Service_Year','Warranty_Expiry_Year'
    ])

    sensor_data = pd.DataFrame([[machine_type, air_temp, process_temp,
        rpm, torque, tool_wear]],
        columns=[
        'Type','Air temperature [K]','Process temperature [K]',
        'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]'
    ])


    maintenance_pred = maintenance_model.predict(vehicle_data)
    failure_pred = failure_model.predict(sensor_data)
    failure_prob = failure_model.predict_proba(sensor_data)

    st.subheader(" Results")

    col3, col4 = st.columns(2)

    with col3:
        if maintenance_pred[0] == 1:
            st.error(" Maintenance Required!")
        else:
            st.success(" Vehicle is Healthy")

    with col4:
        if failure_pred[0] == 1:
            st.error(f" High Failure Risk ({round(failure_prob[0][1]*100,2)}%)")
        else:
            st.success(f" Low Failure Risk ({round(failure_prob[0][1]*100,2)}%)")

    # =======================
    # FEATURE IMPORTANCE
    # =======================

    st.divider()
    st.subheader(" Feature Importance (Failure Model)")

    importance = pd.Series(failure_model.feature_importances_, index=[
    'Type','Air temperature [K]','Process temperature [K]',
    'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]'
    ])

    fig, ax = plt.subplots()
    importance.sort_values().plot(kind='barh', ax=ax)
    st.pyplot(fig)


    # =======================
    #  SENSOR VALUES GRAPH
    # =======================

    st.divider()
    st.subheader(" Sensor Values Overview")

    sensor_df = pd.DataFrame({
    "Feature": ['Air Temp','Process Temp','RPM','Torque','Tool Wear'],
    "Value": [air_temp, process_temp, rpm, torque, tool_wear]
    })

    st.bar_chart(sensor_df.set_index("Feature"))


    # =======================
    #  FAILURE PROBABILITY
    # =======================

    st.divider()
    st.subheader(" Failure Probability")

    prob = round(failure_prob[0][1]*100, 2)

    st.progress(int(prob))

    st.write(f"Failure Risk: {prob}%")