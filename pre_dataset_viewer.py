from src import config

import pandas as pd


df = pd.read_csv(str(config.DATA_PATH))

print(df.head(10))


print(
"""
                      event_hour         source         medium device_type        os users page_views conversions revenue
0 2020-11-01 00:00:00.000000 UTC         google        organic     desktop   Windows     8         22           0     0.0
1 2020-11-01 00:00:00.000000 UTC        <Other>        <Other>     desktop       Web     4        117           0     0.0
2 2020-11-01 00:00:00.000000 UTC       (direct)         (none)     desktop       Web    13         36           0     0.0
3 2020-11-01 00:00:00.000000 UTC        <Other>        <Other>      mobile   Android     5         51           1     0.0
4 2020-11-01 00:00:00.000000 UTC         google        organic     desktop       Web    16         21           0     0.0
5 2020-11-01 00:00:00.000000 UTC         google            cpc     desktop       Web     2          2           0     0.0
6 2020-11-01 00:00:00.000000 UTC (data deleted) (data deleted)      mobile       iOS     1          7           0     0.0
7 2020-11-01 00:00:00.000000 UTC         google        organic     desktop Macintosh     2          3           0     0.0
8 2020-11-01 00:00:00.000000 UTC        <Other>        <Other>      mobile       Web     3          3           0     0.0
9 2020-11-01 00:00:00.000000 UTC         google        organic      mobile       iOS     5         10           0     0.0
"""
)
