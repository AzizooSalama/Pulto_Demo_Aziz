import adi
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# Ensure QApplication is instantiated only if not already done
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication([])

'''Setup'''
samp_rate = 2e6    # must be <=30.72 MHz if both channels are enabled
NumSamples = 2**12
rx_lo = 2.3e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain0 = 40
rx_gain1 = 40
tx_lo = rx_lo
tx_gain = -3
fc0 = int(200e3)
phase_cal = 0
tracking_length = 1000

''' Set distance between Rx antennas '''
d_wavelength = 0.5                  # distance between elements as a fraction of wavelength.  This is normally 0.5
wavelength = 3E8/rx_lo              # wavelength of the RF carrier
d = d_wavelength*wavelength         # distance between elements in meters
print("Set distance between Rx Antennas to ", int(d*1000), "mm")

'''Create Radios'''
sdr1 = adi.Pluto(uri='usb:USB\VID_0456&PID_B673&MI_00\9&fe1711d&0&0000')
sdr2 = adi.Pluto(uri='usb:USB\VID_0456&PID_B673&MI_00\9&81c86bf&0&0000')

'''Configure properties for the Tx Pluto (sdr1)'''
sdr1.tx_enabled_channels = [0, 1]
sdr1.sample_rate = int(samp_rate)
sdr1.tx_rf_bandwidth = int(fc0*3)
sdr1.tx_lo = int(tx_lo)
sdr1.tx_cyclic_buffer = True
sdr1.tx_hardwaregain_chan0 = int(tx_gain)
sdr1.tx_hardwaregain_chan1 = int(-88)  # Set to a low gain as it's not used for transmission

'''Configure properties for the Rx Pluto (sdr2)'''
sdr2.rx_enabled_channels = [0, 1]
sdr2.sample_rate = int(samp_rate)
sdr2.rx_rf_bandwidth = int(fc0*3)
sdr2.rx_lo = int(rx_lo)
sdr2.gain_control_mode = rx_mode
sdr2.rx_hardwaregain_chan0 = int(rx_gain0)
sdr2.rx_hardwaregain_chan1 = int(rx_gain1)
sdr2.rx_buffer_size = int(NumSamples)
sdr2._rxadc.set_kernel_buffers_count(1)   # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto

'''Program Tx and Send Data'''
fs = int(sdr1.sample_rate)
N = 2**16
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
iq0 = i0 + 1j * q0
sdr1.tx([iq0, iq0])  # Send Tx data.

# Assign frequency bins and "zoom in" to the fc0 signal on those frequency bins
xf = np.fft.fftfreq(NumSamples, ts)
xf = np.fft.fftshift(xf) / 1e6
signal_start = int(NumSamples * (samp_rate / 2 + fc0 / 2) / samp_rate)
signal_end = int(NumSamples * (samp_rate / 2 + fc0 * 2) / samp_rate)


def calcTheta(phase):
    # calculates the steering angle for a given phase delta (phase is in deg)
    # steering angle is theta = arcsin(c*deltaphase/(2*pi*f*d)
    arcsin_arg = np.deg2rad(phase) * 3E8 / (2 * np.pi * rx_lo * d)
    arcsin_arg = max(min(1, arcsin_arg), -1)  # arcsin argument must be between 1 and -1, or numpy will throw a warning
    calc_theta = np.rad2deg(np.arcsin(arcsin_arg))
    return calc_theta


def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS
    NumSamples = len(raw_data)
    win = np.hamming(NumSamples)
    y = raw_data * win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20 * np.log10(np.abs(s_shift) / (2 ** 11))  # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_shift, s_dbfs


def monopulse_angle(array1, array2):
    ''' Correlate the sum and delta signals  '''
    # Since our signals are closely aligned in time, we can just return the 'valid' case where the signals completley overlap
    # We can do correlation in the time domain (probably faster) or the freq domain
    # In the time domain, it would just be this:
    # sum_delta_correlation = np.correlate(delayed_sum, delayed_delta, 'valid')
    # But I like the freq domain, because then I can focus just on the fc0 signal of interest
    sum_delta_correlation = np.correlate(array1[signal_start:signal_end], array2[signal_start:signal_end], 'valid')
    angle_diff = np.angle(sum_delta_correlation)
    return angle_diff


def scan_for_DOA():
    # go through all the possible phase shifts and find the peak, that will be the DOA (direction of arrival) aka steer_angle
    data = sdr2.rx()
    Rx_0 = data[0]
    Rx_1 = data[1]
    peak_sum = []
    peak_delta = []
    monopulse_phase = []
    delay_phases = np.arange(-180, 180, 2)  # phase delay in degrees
    for phase_delay in delay_phases:
        delayed_Rx_1 = Rx_1 * np.exp(1j * np.deg2rad(phase_delay + phase_cal))
        delayed_sum = Rx_0 + delayed_Rx_1
        delayed_delta = Rx_0 - delayed_Rx_1
        delayed_sum_fft, delayed_sum_dbfs = dbfs(delayed_sum)
        delayed_delta_fft, delayed_delta_dbfs = dbfs(delayed_delta)
        mono_angle = monopulse_angle(delayed_sum_fft, delayed_delta_fft)

        peak_sum.append(np.max(delayed_sum_dbfs))
        peak_delta.append(np.max(delayed_delta_dbfs))
        monopulse_phase.append(np.sign(mono_angle))

    peak_dbfs = np.max(peak_sum)
    peak_delay_index = np.where(peak_sum == peak_dbfs)
    peak_delay = delay_phases[peak_delay_index[0][0]]
    steer_angle = int(calcTheta(peak_delay))

    return delay_phases, peak_dbfs, peak_delay, steer_angle, peak_sum, peak_delta, monopulse_phase


def Tracking(last_delay):
    # last delay is the peak_delay (in deg) from the last buffer of data collected
    data = sdr2.rx()
    Rx_0 = data[0]
    Rx_1 = data[1]
    delayed_Rx_1 = Rx_1 * np.exp(1j * np.deg2rad(last_delay + phase_cal))
    delayed_sum = Rx_0 + delayed_Rx_1
    delayed_delta = Rx_0 - delayed_Rx_1
    delayed_sum_fft, delayed_sum_dbfs = dbfs(delayed_sum)
    delayed_delta_fft, delayed_delta_dbfs = dbfs(delayed_delta)
    mono_angle = monopulse_angle(delayed_sum_fft, delayed_delta_fft)
    phase_step = 1
    if np.sign(mono_angle) > 0:
        new_delay = last_delay - phase_step
    else:
        new_delay = last_delay + phase_step
    return new_delay


'''Setup Plot Window'''
win = pg.GraphicsLayoutWidget(show=True)
p1 = win.addPlot()
p1.setXRange(-80, 80)
p1.setYRange(0, tracking_length)
p1.setLabel('bottom', 'Steering Angle', 'deg', **{'color': '#FFF', 'size': '14pt'})
p1.showAxis('left', show=False)
p1.showGrid(x=True, alpha=1)
p1.setTitle('Monopulse Tracking:  Angle vs Time', **{'color': '#FFF', 'size': '14pt'})
fn = QtGui.QFont()
fn.setPointSize(15)
p1.getAxis("bottom").setTickFont(fn)

'''Collect Data'''
for i in range(20):
    # let Pluto run for a bit, to do all its calibrations
    data = sdr2.rx()

# scan once to get the direction of arrival (steer_angle) as the initial point for our monopulse tracker
delay_phases, peak_dbfs, peak_delay, steer_angle, peak_sum, peak_delta, monopulse_phase = scan_for_DOA()
delay = peak_delay  # this will be the starting point if we go into tracking mode

tracking_angles = np.zeros(tracking_length)
tracking_angles = tracking_angles + steer_angle
curve1 = p1.plot(tracking_angles, np.arange(tracking_length), pen=pg.mkPen('#FFF', width=3))


def update_tracker():
    global tracking_angles, delay
    delay = Tracking(delay)
    tracking_angles = np.append(tracking_angles, calcTheta(delay))
    tracking_angles = tracking_angles[1:]
    curve1.setData(tracking_angles, np.arange(tracking_length))


timer = pg.QtCore.QTimer()
timer.timeout.connect(update_tracker)
timer.start(0)

# Start Qt event loop unless running in interactive mode or using pyside
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec())
    else:
        app.exec()

sdr1.tx_destroy_buffer()
