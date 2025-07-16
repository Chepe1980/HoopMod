import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import streamlit as st

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
    
    # 1. Polar Plot
    ax1 = fig.add_subplot(231, polar=True)
    ax1.plot(np.radians(theta_vals), hoop, 'r-', label='Hoop Stress', linewidth=2)
    ax1.plot(np.radians(theta_vals), radial, 'b-', label='Radial Stress', linewidth=2)
    ax1.plot(np.radians(theta_vals), shear, 'g-', label='Shear Stress', linewidth=2)
    ax1.set_title('Stress Components (Polar View)', pad=25, fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    
    # 2. Hoop Stress (Cartesian)
    ax2 = fig.add_subplot(232)
    ax2.plot(theta_vals, hoop, 'r-', linewidth=2)
    ax2.set_title('Hoop Stress vs Angle', fontsize=14)
    ax2.set_xlabel('Angle (degrees)', fontsize=12)
    ax2.set_ylabel('Hoop Stress (MPa)', fontsize=12)
    ax2.grid(True)
    
    # 3. All Stress Components
    ax3 = fig.add_subplot(233)
    ax3.plot(theta_vals, hoop, 'r-', label='Hoop Stress', linewidth=2)
    ax3.plot(theta_vals, radial, 'b-', label='Radial Stress', linewidth=2)
    ax3.plot(theta_vals, shear, 'g-', label='Shear Stress', linewidth=2)
    ax3.set_title('All Stress Components', fontsize=14)
    ax3.set_xlabel('Angle (degrees)', fontsize=12)
    ax3.set_ylabel('Stress (MPa)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True)
    
    # 4. 2D Hoop Stress Distribution
    ax4 = fig.add_subplot(234)
    x = np.linspace(-3, 3, 200)  # Increased resolution
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    
    _, Hoop_2d, _ = kirsch_stresses(Sv, Shmin, Shmax, PP, wellbore_pressure, 
                                  np.degrees(Theta), azimuth, dip, deviation, R)
    Hoop_2d[R < 1.0] = np.nan
    
    im = ax4.imshow(Hoop_2d, extent=[-3, 3, -3, 3], origin='lower', 
                   cmap=selected_cmap, aspect='auto')
    ax4.set_title('2D Hoop Stress Distribution (Top View)', fontsize=14)
    ax4.set_xlabel('X (m)', fontsize=12)
    ax4.set_ylabel('Y (m)', fontsize=12)
    cbar = fig.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Hoop Stress (MPa)', fontsize=12)
    
    borehole = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=2)
    ax4.add_patch(borehole)
    
    # 5. 3D Surface Plot
    ax5 = fig.add_subplot(235, projection='3d')
    r_3d = np.linspace(1, 3, 100)  # Increased resolution
    theta_3d = np.radians(np.linspace(0, 360, 100))
    R_3d, Theta_3d = np.meshgrid(r_3d, theta_3d)
    X_3d = R_3d * np.cos(Theta_3d)
    Y_3d = R_3d * np.sin(Theta_3d)
    _, Hoop_3d, _ = kirsch_stresses(Sv, Shmin, Shmax, PP, wellbore_pressure, 
                                  np.degrees(Theta_3d), azimuth, dip, deviation, R_3d)
    
    surf = ax5.plot_surface(X_3d, Y_3d, Hoop_3d, cmap=selected_cmap, 
                           edgecolor='none', antialiased=True)
    ax5.set_title('3D Hoop Stress Distribution', fontsize=14)
    ax5.set_xlabel('X (m)', fontsize=12)
    ax5.set_ylabel('Y (m)', fontsize=12)
    ax5.set_zlabel('Hoop Stress (MPa)', fontsize=12)
    cbar = fig.colorbar(surf, ax=ax5, shrink=0.6, aspect=10)
    cbar.set_label('Hoop Stress (MPa)', fontsize=12)
    
    # 6. Stress Magnitude
    ax6 = fig.add_subplot(236)
    stress_magnitude = np.sqrt(hoop**2 + radial**2 + shear**2)
    ax6.plot(theta_vals, stress_magnitude, 'k-', label='Stress Magnitude', linewidth=2)
    ax6.plot(theta_vals, hoop, 'r-', alpha=0.3, label='Hoop Stress', linewidth=2)
    ax6.set_title('Stress Magnitude', fontsize=14)
    ax6.set_xlabel('Angle (degrees)', fontsize=12)
    ax6.set_ylabel('Stress (MPa)', fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(True)
    
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

def main():
    st.set_page_config(layout="wide")  # Use full page width
    st.title("Borehole Stress Visualization Tool")
    st.markdown("""
    <style>
    .stSlider>div {padding: 0.5rem 0;}
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Input Parameters")
        Sv = st.slider('Sv (MPa)', 0.0, 100.0, 50.0, 1.0)
        Shmin = st.slider('Shmin (MPa)', 0.0, 100.0, 30.0, 1.0)
        Shmax = st.slider('Shmax (MPa)', 0.0, 100.0, 70.0, 1.0)
        PP = st.slider('Pore Pressure (MPa)', 0.0, 100.0, 20.0, 1.0)
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
            df.to_csv('borehole_stresses.csv', index=False)
            st.success("Data exported to 'borehole_stresses.csv'")

    # Main content
    if 'calculate' in st.session_state and st.session_state.calculate:
        update_plots(Sv, Shmin, Shmax, PP, wellbore_pressure, azimuth, dip, deviation, selected_cmap)

if __name__ == "__main__":
    main()
