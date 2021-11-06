import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import scipy.signal as sig
from numpy.fft import fft, ifft, rfft, irfft

directory = "LOSC_Event_tutorial/"

#I copy pasted the read code from simple_read_ligo.py because trying to import it ran that file which gave an error since hdf5 files aren't where it expects them to be
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


#I looked at the plots for the raw data and it seemed like the noise was white
#So I'm calculating the noise here for both Hanford and Livingston on the assumption that the noise is white
def noise_ft(strain):
    strain_ft = rfft((strain) * sig.tukey(len(strain))) #I'm using the tukey window a = 0.5 because it has a flat plateau in the middle 
    ps = strain_ft**2 #take the power spectrum of the strain
    Nft = ps
    #smooth the ps using a gaussian average, I do this by convoluting with a gaussian function
    gauss = sig.windows.gaussian(len(ps),10)
    old_mean = np.mean(Nft)
    for i in range(100):
        Nft = np.abs(irfft(rfft(Nft) * rfft(gauss)))
        Nft = Nft * old_mean / np.mean(Nft) #I rescale using the previous mean because otherwise the Nft balloons after all these smoothing steps
        old_mean = np.mean(Nft)
    Nft = np.append(Nft,[(Nft[-1] + Nft[0]) / 2]) #I had to add an extra entry to the Nft because the last step removes one
    return Nft

def matched_filter(strain, template):
    nft = noise_ft(strain)
    sft = rfft((strain) * sig.windows.tukey(len(strain)))
    tft = rfft((template) * sig.windows.tukey(len(template)))
    sft_white = sft / np.sqrt(nft)
    tft_white = tft / np.sqrt(nft)
    return irfft(sft_white * np.conj(tft_white))


def get_event_signal(mf):
    maxval = np.amax(mf)
    return np.where(mf == maxval)[0][0], maxval**2

def get_event_noise(mf):
    return np.mean(mf**2)

def analyze_event(event_name, hfile, lfile, template):
    strainh,dth,utch = read_file(directory + hfile)
    strainl,dtl,utcl = read_file(directory + lfile)

    th, tl = read_template(directory + template)

    mfh = matched_filter(strainh, th)
    mfl = matched_filter(strainl, tl)

    timeh, signalh = get_event_signal(mfh)
    noiseh = get_event_noise(mfh)
    snrh = signalh / noiseh
    print(event_name + " at Hanford: ", timeh * dth, " SNR: ", snrh)
    timel, signall = get_event_signal(mfl)
    noisel = get_event_noise(mfl)
    snrl = signall / noisel
    print(event_name + " at Livingston: ", timel * dtl, " SNR: ", snrl)
    print(event_name + " average: ", (timeh * dth * snrh + timel * dtl * snrl)/ (snrh + snrl), " SNR: ", ((snrh)**2 + (snrl)**2)/ (snrh + snrl))
    

    plt.clf()
    plt.plot(range(len(mfh)) * dth, mfh, label = "signal time = " + str(timeh * dth))
    plt.title("MF signal for " + event_name + " at Hanford")
    plt.legend()
    plt.xlabel("time")
    plt.savefig(event_name + "_Hanford.png")

    plt.clf()
    plt.plot(range(len(mfl)) * dtl, mfl, label = "signal time = " + str(timel * dtl))
    plt.title("MF signal for " + event_name + " at Livingston")
    plt.legend()
    plt.xlabel("time")
    plt.savefig(event_name + "_Livingston.png")

analyze_event("GW150914","H-H1_LOSC_4_V2-1126259446-32.hdf5","L-L1_LOSC_4_V2-1126259446-32.hdf5","GW150914_4_template.hdf5")
analyze_event("LVT151012","H-H1_LOSC_4_V2-1128678884-32.hdf5","L-L1_LOSC_4_V2-1128678884-32.hdf5","LVT151012_4_template.hdf5")
analyze_event("GW151226","H-H1_LOSC_4_V2-1135136334-32.hdf5","L-L1_LOSC_4_V2-1135136334-32.hdf5","GW151226_4_template.hdf5")
analyze_event("GW170104","H-H1_LOSC_4_V1-1167559920-32.hdf5","L-L1_LOSC_4_V1-1167559920-32.hdf5","GW170104_4_template.hdf5")
    
    
