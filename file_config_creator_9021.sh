#!/bin/bash

# Définir le chemin du fichier config.txt
config_file="config_9021.txt"

# Liste des variables
variables=(
    "10m_u_component_of_wind"
    "10m_v_component_of_wind"
    "2m_dewpoint_temperature"
    "2m_temperature"
    "cloud_base_height"
    "convective_available_potential_energy"
    "convective_inhibition"
    "convective_precipitation"
    "convective_rain_rate"
    "convective_snowfall"
    "high_cloud_cover"
    "instantaneous_10m_wind_gust"
    "k_index"
    "land_sea_mask"
    "large_scale_precipitation"
    "large_scale_snowfall"
    "mean_large_scale_precipitation_rate"
    "mean_top_net_long_wave_radiation_flux"
    "mean_top_net_short_wave_radiation_flux"
    "mean_total_precipitation_rate"
    "mean_sea_level_pressure"
    "mean_surface_latent_heat_flux"
    "mean_surface_net_long_wave_radiation_flux"
    "mean_surface_net_short_wave_radiation_flux"
    "mean_vertically_integrated_moisture_divergence"
    "surface_pressure"
    "total_precipitation"
    "total_totals_index"
)

# Années 1990 et 2021
years=(1990 2021)

# Supprimer le fichier config.txt s'il existe déjà
rm -f "$config_file"

# Écrire l'en-tête du fichier config.txt
echo "ArrayTaskID  Nom_dossier  Année" >> "$config_file"

# Parcourir chaque variable et chaque année pour créer les lignes dans le fichier config.txt
task_id=1
for variable in "${variables[@]}"
do
    for year in "${years[@]}"
    do
        echo "$task_id  $variable  $year" >> "$config_file"
        ((task_id++))
    done
done

echo "Le fichier $config_file a été créé avec succès."
