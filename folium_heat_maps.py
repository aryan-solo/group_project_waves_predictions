from folium import plugins
from folium.plugins import HeatMap
import pandas as pd
import folium
from datetime import datetime, timedelta
from folium.features import DivIcon

coordinate_map = {"M1":[-11.2,53.1266],
                  "Belmullet-AMETS":[-10.14221,54.2659],
                  "FS1":[-7.9,51.4],
                  "M4-Archive":[-9.0667,54.6667],
                  "M2":[-5.4302,53.4836],
                  "M3":[-10.548261,51.215956],
                  "M4":[-9.999136,54.999967],
                  "M5":[-6.704336,51.690425],
                  "M6":[-15.88135,53.07482]
                  }


m = folium.Map([53.523611, -7.754379], zoom_start=6)

data = []
df_power = pd.read_csv('./data/Power_Aquabuoy_month_MWh.csv')#,parse_dates=["date"],index_col=["date"])
df_power = df_power.fillna(0)

print(df_power.head())

st = ['M1','M2','M3','M4','M6','FS1']
time_index = []

for o in st:
    #folium.Circle([coordinate_map[o][1],coordinate_map[o][0]], 30000, fill=False).add_child(folium.Popup(o)).add_to(m)
    folium.map.Marker(
    [coordinate_map[o][1] + 0.5, coordinate_map[o][0] - 1.6],
    icon=DivIcon(
        icon_size=(150,36),
        icon_anchor=(0,0),
        html='<div style="font-size: 24pt">%s</div>' % o,
        )
    ).add_to(m)

for index, row in df_power.iterrows():
    time_index.append(row[0])
    d = {
        'M1' : row[1],
        'M2' : row[2],
        'M3' : row[3],
        'M4' : row[4],
        'M6' : row[5],
        'FS1' : row[6]
        }
    for one in st:
        d1 = []
        if d[one] < 1:
            continue
        time_index.append(row[0])
        d1 = [coordinate_map[one][1],coordinate_map[one][0]]
        data.append(d1)



HeatMap(data).add_to(m)

#folium.LayerControl().add_to(m)

m.save("test.html")



