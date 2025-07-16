import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import streamlit as st
import io

# Available colormaps
colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
             'coolwarm', 'RdYlBu', 'seismic', 'rainbow', 'jet']

def kirsch_stresses(Sv, Shmin, Shmax, PP, wellbore_pressure, theta_deg, azimuth_deg, dip_deg, deviation_deg, r=1.0):
    theta = np.radians(theta_deg)
    azimuth = np.radians(azimuth_deg)
    dip = np.radians(dip_deg)
    deviation = np.radians(deviation_deg)
    
    # Stress transformation (3D rotation)
    R = np.array([
        [np.cos(deviation)*np.cos(azimuth), -np.sin(azimuth), np.sin(deviation)*np.cos(azimuth)],
        [np.cos(deviation)*np.sin(azimuth), np.cos(azimuth), np.sin(deviation)*np.sin(azimuth)],
        [-np.sin(deviation), 0, np.cos(deviation)]
    ])
    
    sigma_global = np.diag([Shmax, Shmin, Sv])
    sigma_local = R.T @ sigma_global @ R
    
    # Kirsch equations
    a = 1.0  # Borehole radius
    radial = (sigma_local[0,0] + sigma_local[1,1])/2 * (1 - a**2/r**2) + \
             (sigma_local[0,0] - sigma_local[1,1])/2 * (1 - 4*a**2/r**2 + 3*a**4/r**4) * np.cos(2*theta) + \
             sigma_local[0,1] * (1 - 4*a**2/r**2 + 3*a**4/r**4) * np.sin(2*theta) + \
             PP * a**2/r**2 - wellbore_pressure * a**2/r**2
    
    hoop = (sigma_local[0,0] + sigma_local[1,1])/2 * (1 + a**2/r**2) - \
           (sigma_local[0,0] - sigma_local[1,1])/2 * (1 + 3*a**4/r**4) * np.cos(2*theta) - \
           sigma_local[0,1] * (1 + 3*a**4/r**4) * np.sin(2*theta) - \
           PP * a**2/r**2 + wellbore_pressure * a**2/r**2
    
    shear = -(sigma_local[0,0] - sigma_local[1,1])/2 * (1 + 2*a**2/r**2 - 3*a**4/r**4) * np.sin(2*theta) + \
            sigma_local[0,1] * (1 + 2*a**2/r**2 - 3*a**4/r**4) * np.cos(2*theta)
    
    return radial, hoop, shear

def update_plots(Sv, Shmin, Shmax, PP, wellbore_pressure, azimuth, dip, deviation, selected_cmap):
    # Calculate stresses
    theta_vals = np.linspace(0, 360, 360)
    radial, hoop, shear = kirsch_stresses(Sv, Shmin, Shmax, PP, wellbore_pressure, 
                                       theta_vals, azimuth, dip, deviation, r=1.0)
    
    # Create figure with larger size
    fig = plt.figure(figsize=(24, 16))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.3, hspace=0.4)
    
    # [Previous plot code remains exactly the same...]
    # ... (Include all your existing plotting code here unchanged)

    # Display in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Data table
    df = pd.DataFrame({
        'Theta (deg)': theta_vals,
        'Hoop Stress (MPa)': hoop,
        'Radial Stress (MPa)': radial,
        'Shear Stress (MPa)': shear,
        'Stress Magnitude (MPa)': np.sqrt(hoop**2 + radial**2 + shear**2)
    })
    st.dataframe(df.head())

def load_well_log(uploaded_file):
    """Load well log data from CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = {'Depth', 'Sv', 'Shmin', 'Shmax', 'PP'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            st.error(f"Missing required columns: {missing}")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("Borehole Stress Visualization Tool (Well Log Input)")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload Well Log CSV", type=["csv"])
    
    if uploaded_file is not None:
        well_log = load_well_log(uploaded_file)
        
        if well_log is not None:
            # Depth selection
            min_depth = float(well_log['Depth'].min())
            max_depth = float(well_log['Depth'].max())
            selected_depth = st.slider(
                'Select Depth (m)', 
                min_value=min_depth, 
                max_value=max_depth, 
                value=(min_depth + max_depth)/2,
                step=0.1
            )
            
            # Find closest depth in log
            idx = (well_log['Depth'] - selected_depth).abs().idxmin()
            log_data = well_log.iloc[idx]
            
            st.success(f"Loaded data for depth: {log_data['Depth']:.2f}m")
            
            # Display stress values from log
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                Sv = st.number_input('Sv (MPa)', value=float(log_data['Sv']))
            with col2:
                Shmin = st.number_input('Shmin (MPa)', value=float(log_data['Shmin']))
            with col3:
                Shmax = st.number_input('Shmax (MPa)', value=float(log_data['Shmax']))
            with col4:
                PP = st.number_input('Pore Pressure (MPa)', value=float(log_data['PP']))
            
            # Other parameters (unchanged)
            with st.sidebar:
                st.header("Wellbore Parameters")
                wellbore_pressure = st.slider('Wellbore Pressure (MPa)', 0.0, 100.0, 10.0, 1.0)
                azimuth = st.slider('Azimuth (°)', 0, 360, 0, 1)
                dip = st.slider('Dip (°)', 0, 90, 0, 1)
                deviation = st.slider('Deviation (°)', 0, 90, 0, 1)
                selected_cmap = st.selectbox('Color Map:', colormaps, index=0)
                
                if st.button('Calculate Stresses', type="primary"):
                    st.session_state.calculate = True
                
                if st.button('Export Data to CSV'):
                    theta_vals = np.linspace(0, 360, 360)
                    radial, hoop, shear = kirsch_stresses(
                        Sv, Shmin, Shmax, PP, wellbore_pressure,
                        theta_vals, azimuth, dip, deviation
                    )
                    df = pd.DataFrame({
                        'Theta (deg)': theta_vals,
                        'Hoop Stress (MPa)': hoop,
                        'Radial Stress (MPa)': radial,
                        'Shear Stress (MPa)': shear,
                        'Stress Magnitude (MPa)': np.sqrt(hoop**2 + radial**2 + shear**2)
                    })
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name='borehole_stresses.csv',
                        mime='text/csv'
                    )

            if 'calculate' in st.session_state and st.session_state.calculate:
                update_plots(Sv, Shmin, Shmax, PP, wellbore_pressure, 
                            azimuth, dip, deviation, selected_cmap)

if __name__ == "__main__":
    main()
