import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import scipy.signal as sig
from numpy.fft import fft, ifft, rfft, irfft

directory = "LOSC_Event_tutorial/"

def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl
def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc



def noise_ft(strain):
    strain_ft = rfft((strain) * sig.tukey(len(strain))) #I'm using the tukey window a = 0.5 because it has a flat plateau in the middle 
    ps = strain_ft**2 #take the power spectrum of the strain
    Nft = ps
    #smooth the ps using a gaussian average, I do this by convoluting with a gaussian function
    gauss = sig.windows.gaussian(len(ps),10)
    old_mean = np.mean(Nft)
    for i in range(100):
        Nft = np.abs(irfft(rfft(Nft) * rfft(gauss)))
        Nft = Nft * old_mean / np.mean(Nft)
        old_mean = np.mean(Nft)
        #Nft = (Nft+np.roll(Nft,1)+np.roll(Nft,-1))/3
    Nft = np.append(Nft,[(Nft[-1] + Nft[0]) / 2])
    #plt.plot(ps)
    return Nft

#plt.plot(ifft(fft(strain1) / np.sqrt(noise_ft(strain1))))
#plt.plot(ifft(fft(tl) / np.sqrt(noise_ft(strain2))))
#plt.plot(noise_ft(strain1))

def matched_filter(strain, template):
    nft = noise_ft(strain)
    sft = rfft((strain) * sig.windows.tukey(len(strain)))
    tft = rfft((template) * sig.windows.tukey(len(template)))
    sft_white = sft / np.sqrt(nft)
    tft_white = tft / np.sqrt(nft)
    return irfft(sft_white * np.conj(tft_white))

#plt.plot(matched_filter(strain2,tl))
#plt.plot(sig.windows.hann(len(strain1)))

def get_event_signal(mf):
    maxval = np.amax(mf)
    return np.where(mf == maxval)[0][0], np.abs(maxval)

def get_event_noise(mf):
    return np.mean(np.abs(mf))

def analyze_event(event_name, hfile, lfile, template):
    strainh,dth,utch = read_file(directory + hfile)
    strainl,dtl,utcl = read_file(directory + lfile)

    th, tl = read_template(directory + template)

    mfh = matched_filter(strainh, th)
    mfl = matched_filter(strainl, tl)

    timeh, signalh = get_event_signal(mfh)
    noiseh = get_event_noise(mfh)
    print(event_name + " at Hanford: ", timeh * dth, " SNR: ", signalh / noiseh)
    timel, signall = get_event_signal(mfl)
    noisel = get_event_noise(mfl)
    print(event_name + " at Livingston: ", timel * dtl, " SNR: ", signall / noisel)

    plt.clf()
    plt.plot(mfh)
    plt.plot(mfl)
    plt.savefig(event_name + ".png")

analyze_event("GW150914","H-H1_LOSC_4_V2-1126259446-32.hdf5","L-L1_LOSC_4_V2-1126259446-32.hdf5","GW150914_4_template.hdf5")
analyze_event("LVT151012","H-H1_LOSC_4_V2-1128678884-32.hdf5","L-L1_LOSC_4_V2-1128678884-32.hdf5","LVT151012_4_template.hdf5")
analyze_event("GW151226","H-H1_LOSC_4_V2-1135136334-32.hdf5","L-L1_LOSC_4_V2-1135136334-32.hdf5","GW151226_4_template.hdf5")
analyze_event("GW170104","H-H1_LOSC_4_V1-1167559920-32.hdf5","L-L1_LOSC_4_V1-1167559920-32.hdf5","GW170104_4_template.hdf5")
    
    
