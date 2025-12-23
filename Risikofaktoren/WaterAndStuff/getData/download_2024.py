import xarray as xr
import pandas as pd
import numpy as np
import os
import glob
from datetime import timedelta

grib_folder = r"C:\Users\richt\Desktop\testNaturgefahrenOhneBuffer\GRIB_DATA"
csv_path = r"C:\Users\richt\Desktop\testNaturgefahrenOhneBuffer\landslides_WITH_NEW_RAIN.csv"

def main():
    
    # CSV laden
    df = pd.read_csv(csv_path)
    df['Datum'] = pd.to_datetime(df['Datum'])
    
    # Lücken finden (Nur 2024)
    mask = (df['Rainfall_7d_mm'].isna()) & (df['Datum'].dt.year == 2024)
    missing_indices = df[mask].index
    
    if len(missing_indices) == 0:
        print("Keine Lücken in 2024 mehr.")
        return

    print(f"Muss noch {len(missing_indices)} Fälle reparieren...")

    # Cache
    loaded_gribs = {}

    # Funktion: Findet irgendeine Datei, die diesen Tag abdeckt
    def find_valid_file(target_date):
        # geht bis zu 4 Tage zurück
        for i in range(5): 
            check_date = target_date - timedelta(days=i)
            fname = f"MERIDA_PREC_{check_date.strftime('%Y%m%d')}_00.grb2"
            full_path = os.path.join(grib_folder, fname)
            
            # Datei muss existieren UND größer als 10KB sein
            if os.path.exists(full_path) and os.path.getsize(full_path) > 10000:
                # Prüfen, ob der Zeitabstand im Rahmen ist (0 bis 72h)
                hours_diff = (target_date - check_date).days * 24
                # Manche Dateien haben vielleicht mehr als 72h? -> alles bis 96h zur Sicherheit
                if 0 <= hours_diff <= 96: 
                    return full_path, hours_diff
        return None, None

    # Loop über Lücken
    for idx in missing_indices:
        row = df.loc[idx]
        event_date = row['Datum']
        lat, lon = row['Latitude'], row['Longitude']
        
        print(f"Repariere Zeile {idx} ({event_date.date()})...")
        
        # Berechnung für 7, 14, 21 Tage
        for duration in [7, 14, 21]:
            rain_sum = 0
            valid_days = 0
            
            # Jeden Tag einzeln durchgehen
            for day_offset in range(duration):
                current_lookback_date = event_date - timedelta(days=day_offset)
                
                # Finde Datei für diesen spezifischen Tag
                grib_path, hour_start_of_day = find_valid_file(current_lookback_date)
                
                if grib_path:
                    # Laden (mit Cache)
                    if grib_path not in loaded_gribs:
                        try:
                            # Index vorher löschen
                            for idx_file in glob.glob(grib_path + "*.idx"):
                                try: os.remove(idx_file)
                                except: pass
                            
                            ds = xr.open_dataset(grib_path, engine="cfgrib", 
                                                 backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
                            loaded_gribs[grib_path] = ds
                        except Exception as e:
                            print(f"  Defekte Datei übersprungen: {os.path.basename(grib_path)}")
                            continue
                    
                    ds = loaded_gribs[grib_path]
                    
                    try:
                        # Regen berechnen (Differenz zwischen Ende des Tages und anfang des Tages)
                        # hour_start_of_day ist z.B. 24 (Beginn des 2. Tages im File)
                        # Wert bei 48h minus Wert bei 24h
                        h_start = hour_start_of_day
                        h_end = hour_start_of_day + 24
                        
                        # Nearest Neighbor
                        point = ds['tp'].sel(latitude=lat, longitude=lon, method='nearest')
                        vals = point.values
                        steps = ds['step'].values.astype('timedelta64[h]').astype(int)
                        
                        val_start = 0
                        val_end = 0
                        
                        # Suche Index für Startzeit
                        if h_start > 0:
                            idx_s = np.where(steps == h_start)[0]
                            if len(idx_s) > 0: val_start = vals[idx_s[0]]
                        
                        # Suche Index für Endzeit
                        idx_e = np.where(steps == h_end)[0]
                        if len(idx_e) > 0: 
                            val_end = vals[idx_e[0]]
                            
                            # Tag nur gültig wenn mit endwert
                            day_rain = max(0, val_end - val_start)
                            rain_sum += day_rain
                            valid_days += 1
                            
                    except Exception:
                        pass # Einzelner Tag fehlgeschlagen
            
            # Eintragen wenn genug Daten da
            if valid_days > (duration / 2):
                col = f'Rainfall_{duration}d_mm'
                df.at[idx, col] = round(rain_sum, 2)
                print(f"  -> {col}: {round(rain_sum, 2)}")

    df.to_csv(csv_path, index=False)
    print("Fertig")

if __name__ == "__main__":
    main()