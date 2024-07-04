import adi
import matplotlib.pyplot as plt
import numpy as np

'''Setup'''
samp_rate = 2e6  # must be <=30.72 MHz if both channels are enabled
NumSamples = 2 ** 12
rx_lo = 2.3e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain0 = 40
tx_lo = rx_lo
tx_gain = -3
fc0 = int(250e3)
num_scans = 10000

'''Create Radios'''
sdr = adi.Pluto(uri='usb:')
sdr1_tx = adi.Pluto(uri='usb:USB\VID_0456&PID_B673&MI_00\9&fe1711d&0&0000')



'''Configure properties for the Rx Pluto'''
sdr.rx_enabled_channels = [0]
sdr.sample_rate = int(samp_rate)
sdr.rx_rf_bandwidth = int(fc0 * 3)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode = rx_mode
sdr.rx_hardwaregain_chan0 = int(rx_gain0)
sdr.rx_buffer_size = int(NumSamples)
sdr._rxadc.set_kernel_buffers_count(1)  # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto
sdr.tx_rf_bandwidth = int(fc0 * 3)
sdr.tx_lo = int(rx_lo)
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = int(tx_gain)
sdr.tx_buffer_size = int(2 ** 18)

'''Program Tx and Send Data'''
fs = int(sdr.sample_rate)
N = 2 ** 16
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
iq0 = i0 + 1j * q0
sdr.tx(iq0)  # Send Tx data.

# Assign frequency bins and "zoom in" to the fc0 signal on those frequency bins
xf = np.fft.fftfreq(NumSamples, ts)
xf = np.fft.fftshift(xf) / 1e6
signal_start = int(NumSamples * (samp_rate / 2 + fc0 / 2) / samp_rate)
signal_end = int(NumSamples * (samp_rate / 2 + fc0 * 2) / samp_rate)


def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS
    NumSamples = len(raw_data)
    win = np.hamming(NumSamples)
    y = raw_data * win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20 * np.log10(np.abs(s_shift) / (2 ** 11))  # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_shift, s_dbfs


def scan_for_DOA():
    data = sdr.rx()
    print("Type of data:", type(data))
    print("Data shape:", data.shape if isinstance(data, np.ndarray) else "N/A")
    print("First element of data:", data[0] if isinstance(data, np.ndarray) else "N/A")

    delayed_sum_fft, delayed_sum_dbfs = dbfs(data)

    peak_dbfs = np.max(delayed_sum_dbfs)

    return peak_dbfs, delayed_sum_dbfs


'''Collect Data'''
for i in range(20):
    # let Pluto run for a bit, to do all its calibrations
    data = sdr.rx()

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot(xf, np.zeros_like(xf))
ax.axvline(x=fc0 / 1e6, color='r', linestyle=':')
ax.text(-samp_rate / 2 / 1e6, -26, "Peak signal occurs at {} MHz".format(round(fc0 / 1e6, 2)))
ax.set_ylim(top=20, bottom=-120)
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("Amplitude [dBFS]")
plt.show()  # Ensure the initial plot is rendered

for i in range(num_scans):
    peak_dbfs, delayed_sum_dbfs = scan_for_DOA()

    line.set_ydata(delayed_sum_dbfs)  # Update y data of the plot
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)  # Pause to update the plot

plt.ioff()  # Turn off interactive mode

sdr.tx_destroy_buffer()
if i > 40:
    print('\a')  # for a long capture, beep when the script is done
