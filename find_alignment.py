# %%
import os
import getpass
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"
username = getpass.getuser()
print(f'Assuming that your kerberos keytab is in the home folder, '
      f'its name is "{getpass.getuser()}.keytab" '
      f'and that your kerberos login is "{username}".')

logging.info('Executing the kinit')
os.system(f'kinit -f -r 5d -kt {os.path.expanduser("~")}/'+
          f'{getpass.getuser()}.keytab {getpass.getuser()}');

# %%
import json
import nx2pd as nx
import pandas as pd

from nxcals.spark_session_builder import get_or_create, Flavor

logging.info('Creating the spark instance')

# Here I am using YARN (to do compution on the cluster)
#spark = get_or_create(flavor=Flavor.YARN_SMALL, master='yarn')

# Here I am using the LOCAL (assuming that my data are small data,
# so the overhead of the YARN is not justified)
# WARNING: the very first time you need to use YARN
# spark = get_or_create(flavor=Flavor.LOCAL)

logging.info('Creating the spark instance')
spark = get_or_create(flavor=Flavor.LOCAL,
conf={'spark.driver.maxResultSize': '8g',
    'spark.executor.memory':'8g',
    'spark.driver.memory': '16g',
    'spark.executor.instances': '20',
    'spark.executor.cores': '2',
    })
sk  = nx.SparkIt(spark)
logging.info('Spark instance created.')


# %%
t0 = pd.Timestamp('2024-03-30 19:50',tz="CET")
t1 = pd.Timestamp('2024-03-30 21:00',tz="CET")

df = sk.get(t0, t1, [
                    'TCTPH.4L5.B1:MEAS_V_LVDT_POS',
                    'TCTPH.4R5.B2:MEAS_V_LVDT_POS',
                    'TCTPV.4L1.B1:MEAS_V_LVDT_POS',
                    'TCTPV.4R1.B2:MEAS_V_LVDT_POS',
                    'LHC.BPTUV.A4L1.B1:CALIBRAWVALV1',
                    'LHC.BPTUV.A4L1.B1:CALIBRAWVALV2',
                    'LHC.BPTUV.A4R1.B2:CALIBRAWVALV1',
                    'LHC.BPTUV.A4R1.B2:CALIBRAWVALV2',
                    'LHC.BPTUH.A4L5.B1:CALIBRAWVALV1',
                    'LHC.BPTUH.A4L5.B1:CALIBRAWVALV2',
                    'LHC.BPTUH.A4R5.B2:CALIBRAWVALV1',
                    'LHC.BPTUH.A4R5.B2:CALIBRAWVALV2',
                    'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY',
                    'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY',
                     ])
# iterpolate all the missing values in the dataframe
# %%
df = df.interpolate(method='time')
df['DCT_B1_FIT'] = df['LHC.BCTDC.A6R4.B1:BEAM_INTENSITY'].rolling(window=5000).mean()
df['DCT_B2_FIT'] = df['LHC.BCTDC.A6R4.B2:BEAM_INTENSITY'].rolling(window=5000).mean()

# %%
df.to_parquet('data/df.csv')

# %%
from matplotlib import pyplot as plt
t0_filtered="19:28"
t1_filtered="19:36"

wire = 'L1'
if 'L' in wire:
    my_string = wire + '.B1'
    my_beam = 'B1'
else:
    my_string = wire + '.B2'
    my_beam = 'B2'
if '1' in wire:
    my_plane = 'V'
else:
    my_plane = 'H'
    
initial_offset = df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered)[0]

plt.plot(df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered), label=f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS')
plt.axhline(y=initial_offset, color='r', linestyle='--')
plt.xticks(rotation=45)
plt.grid()
plt.title(f'Initial offset {initial_offset:3.2f} mm')
plt.ylabel(f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS [mm]')
# plot on the secondary y-axis the LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
plt.twinx()
plt.plot(df[f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY'].between_time(t0_filtered,t1_filtered), label=f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY', color='k',alpha=0.1)
plt.ylabel(f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY [p]')
plt.plot(df['DCT_B1_FIT'].between_time(t0_filtered,t1_filtered), label=f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY', color='k')
plt.savefig(f'plots/scan_TCTP{my_plane}.4{my_string}.png')
# %%
plt.plot(df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered),
          df[f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV1'].between_time(t0_filtered,t1_filtered)/
         df['DCT_B1_FIT'].between_time(t0_filtered,t1_filtered),'.-', label=f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV1')
plt.plot(df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered),
          df[f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV2'].between_time(t0_filtered,t1_filtered)/
          df['DCT_B1_FIT'].between_time(t0_filtered,t1_filtered),'.-', label=f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV2')
plt.xticks(rotation=45)
plt.grid()
plt.ylabel('[arb. units]')
plt.xlabel('TCTPV.4L1.B1:MEAS_V_LVDT_POS [mm]')
plt.legend()
# plot vertical line at the 'LHC.BPTUV.A4L1.B1:CALIBRAWVALV1' maximum
initial_offset = df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered).dropna()[0]
offset = 0.
plt.axvline(x=initial_offset, color='r', linestyle='--')
plt.axvline(x=offset, color='g', linestyle='--')

plt.title(f'Align from {initial_offset:3.2f} to {offset} mm')
plt.savefig(f'plots/result_TCTP{my_plane}.4{my_string}.png')

# %%
t0_filtered="19:37"
t1_filtered="19:47"

wire = 'R1'
if 'L' in wire:
    my_string = wire + '.B1'
    my_beam = 'B1'
else:
    my_string = wire + '.B2'
    my_beam = 'B2'
if '1' in wire:
    my_plane = 'V'
else:
    my_plane = 'H'
    
initial_offset = df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered)[0]

plt.plot(df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered), label=f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS')
plt.axhline(y=initial_offset, color='r', linestyle='--')
plt.xticks(rotation=45)
plt.grid()
plt.title(f'Initial offset {initial_offset:3.2f} mm')
plt.ylabel(f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS [mm]')
# plot on the secondary y-axis the LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
plt.twinx()
plt.plot(df[f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY'].between_time(t0_filtered,t1_filtered), label=f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY', color='k',alpha=0.1)
plt.ylabel(f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY [p]')
plt.plot(df['DCT_B1_FIT'].between_time(t0_filtered,t1_filtered), label=f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY', color='k')
plt.savefig(f'plots/scan_TCTP{my_plane}.4{my_string}.png')
# %%
plt.plot(df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered),
          df[f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV1'].between_time(t0_filtered,t1_filtered)/
         df['DCT_B1_FIT'].between_time(t0_filtered,t1_filtered),'.-', label=f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV1')
plt.plot(df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered),
          df[f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV2'].between_time(t0_filtered,t1_filtered)/
          df['DCT_B1_FIT'].between_time(t0_filtered,t1_filtered),'.-', label=f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV2')
plt.xticks(rotation=45)
plt.grid()
plt.ylabel('[arb. units]')
plt.xlabel('TCTPV.4L1.B1:MEAS_V_LVDT_POS [mm]')
plt.legend()
# plot vertical line at the 'LHC.BPTUV.A4L1.B1:CALIBRAWVALV1' maximum
initial_offset = df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered).dropna()[0]
offset = 0.3
plt.axvline(x=initial_offset, color='r', linestyle='--')
plt.axvline(x=offset, color='g', linestyle='--')

plt.title(f'Align from {initial_offset:3.2f} to {offset} mm')
plt.savefig(f'plots/result_TCTP{my_plane}.4{my_string}.png')

# %%
t0_filtered="19:08"
t1_filtered="19:18"

wire = 'L5'
if 'L' in wire:
    my_string = wire + '.B1'
    my_beam = 'B1'
else:
    my_string = wire + '.B2'
    my_beam = 'B2'
if '1' in wire:
    my_plane = 'V'
else:
    my_plane = 'H'
    
initial_offset = df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered)[0]

plt.plot(df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered), label=f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS')
plt.axhline(y=initial_offset, color='r', linestyle='--')
plt.xticks(rotation=45)
plt.grid()
plt.title(f'Initial offset {initial_offset:3.2f} mm')
plt.ylabel(f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS [mm]')
# plot on the secondary y-axis the LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
plt.twinx()
plt.plot(df[f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY'].between_time(t0_filtered,t1_filtered), label=f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY', color='k',alpha=0.1)
plt.ylabel(f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY [p]')
plt.plot(df['DCT_B1_FIT'].between_time(t0_filtered,t1_filtered), label=f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY', color='k')
plt.savefig(f'plots/scan_TCTP{my_plane}.4{my_string}.png')
# %%
plt.plot(df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered),
          df[f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV1'].between_time(t0_filtered,t1_filtered)/
         df['DCT_B1_FIT'].between_time(t0_filtered,t1_filtered),'.-', label=f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV1')
plt.plot(df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered),
          df[f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV2'].between_time(t0_filtered,t1_filtered)/
          df['DCT_B1_FIT'].between_time(t0_filtered,t1_filtered),'.-', label=f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV2')
plt.xticks(rotation=45)
plt.grid()
plt.ylabel('[arb. units]')
plt.xlabel('TCTPV.4L1.B1:MEAS_V_LVDT_POS [mm]')
plt.legend()
# plot vertical line at the 'LHC.BPTUV.A4L1.B1:CALIBRAWVALV1' maximum
initial_offset = df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered).dropna()[0]
offset = 1.2
plt.axvline(x=initial_offset, color='r', linestyle='--')
plt.axvline(x=offset, color='g', linestyle='--')

plt.title(f'Align from {initial_offset:3.2f} to {offset} mm')
plt.savefig(f'plots/result_TCTP{my_plane}.4{my_string}.png')
# %%
t0_filtered="19:27"
t1_filtered="19:37"

wire = 'L1'
if 'L' in wire:
    my_string = wire + '.B1'
    my_beam = 'B1'
else:
    my_string = wire + '.B2'
    my_beam = 'B2'
if '1' in wire:
    my_plane = 'V'
else:
    my_plane = 'H'
    
initial_offset = df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered)[0]

plt.plot(df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered), label=f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS')
plt.axhline(y=initial_offset, color='r', linestyle='--')
plt.xticks(rotation=45)
plt.grid()
plt.title(f'Initial offset {initial_offset:3.2f} mm')
plt.ylabel(f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS [mm]')
# plot on the secondary y-axis the LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
plt.twinx()
plt.plot(df[f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY'].between_time(t0_filtered,t1_filtered), label=f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY', color='k',alpha=0.1)
plt.ylabel(f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY [p]')
plt.plot(df['DCT_B1_FIT'].between_time(t0_filtered,t1_filtered), label=f'LHC.BCTDC.A6R4.{my_beam}:BEAM_INTENSITY', color='k')
plt.savefig(f'plots/scan_TCTP{my_plane}.4{my_string}.png')
# %%
plt.plot(df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered),
          df[f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV1'].between_time(t0_filtered,t1_filtered)/
         df['DCT_B1_FIT'].between_time(t0_filtered,t1_filtered),'.-', label=f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV1')
plt.plot(df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered),
          df[f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV2'].between_time(t0_filtered,t1_filtered)/
          df['DCT_B1_FIT'].between_time(t0_filtered,t1_filtered),'.-', label=f'LHC.BPTU{my_plane}.A4{my_string}:CALIBRAWVALV2')
plt.xticks(rotation=45)
plt.grid()
plt.ylabel('[arb. units]')
plt.xlabel('TCTPV.4L1.B1:MEAS_V_LVDT_POS [mm]')
plt.legend()
# plot vertical line at the 'LHC.BPTUV.A4L1.B1:CALIBRAWVALV1' maximum
initial_offset = df[f'TCTP{my_plane}.4{my_string}:MEAS_V_LVDT_POS'].between_time(t0_filtered,t1_filtered).dropna()[0]
offset = 0
plt.axvline(x=initial_offset, color='r', linestyle='--')
plt.axvline(x=offset, color='g', linestyle='--')

plt.title(f'Align from {initial_offset:3.2f} to {offset} mm')
plt.savefig(f'plots/result_TCTP{my_plane}.4{my_string}.png')
# %%
