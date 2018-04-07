import numpy as np
import scipy

"""
Code referenced from: https://github.com/lonce/SPSI_Python
"""


def spsi(msgram, fftsize, hop_length) :
    """
    Takes a 2D spectrogram ([freqs,frames]), the fft legnth (= widnow length) and the hope size (both in units of samples).
    Returns an audio signal.
    """
    
    numBins, numFrames  = msgram.shape
    y_out=np.zeros(numFrames*hop_length+fftsize-hop_length)
        
    m_phase=np.zeros(numBins);      
    m_win=scipy.signal.hanning(fftsize, sym=True)  # assumption here that hann was used to create the frames of the spectrogram
    
    #processes one frame of audio at a time
    for i in range(numFrames) :
            m_mag=msgram[:, i] 
            for j in range(1,numBins-1) : 
                if(m_mag[j]>m_mag[j-1] and m_mag[j]>m_mag[j+1]) : #if j is a peak
                    alpha=m_mag[j-1];
                    beta=m_mag[j];
                    gamma=m_mag[j+1];
                    denom=alpha-2*beta+gamma;
                    
                    if(denom!=0) :
                        p=0.5*(alpha-gamma)/denom;
                    else :
                        p=0;
                        
                    phaseRate=2*np.pi*(j+p)/fftsize;    #adjusted phase rate
                    m_phase[j]= m_phase[j] + hop_length*phaseRate; #phase accumulator for this peak bin
                    peakPhase=m_phase[j];
                    
                    # If actual peak is to the right of the bin freq
                    if (p>0) :
                        # First bin to right has pi shift
                        bin=j+1;
                        m_phase[bin]=peakPhase+np.pi;
                        
                        # Bins to left have shift of pi
                        bin=j-1;
                        while((bin>1) and (m_mag[bin]<m_mag[bin+1])) : # until you reach the trough
                            m_phase[bin]=peakPhase+np.pi;
                            bin=bin-1;
                        
                        #Bins to the right (beyond the first) have 0 shift
                        bin=j+2;
                        while((bin<(numBins)) and (m_mag[bin]<m_mag[bin-1])) :
                            m_phase[bin]=peakPhase;
                            bin=bin+1;
                            
                    #if actual peak is to the left of the bin frequency
                    if(p<0) :
                        # First bin to left has pi shift
                        bin=j-1;
                        m_phase[bin]=peakPhase+np.pi;

                        # and bins to the right of me - here I am stuck in the middle with you
                        bin=j+1;
                        while((bin<(numBins)) and (m_mag[bin]<m_mag[bin-1])) :
                            m_phase[bin]=peakPhase+np.pi;
                            bin=bin+1;
                        
                        # and further to the left have zero shift
                        bin=j-2;
                        while((bin>1) and (m_mag[bin]<m_mag[bin+1])) : # until trough
                            m_phase[bin]=peakPhase;
                            bin=bin-1;
                            
                #end ops for peaks
            #end loop over fft bins with

            magphase=m_mag*np.exp(1j*m_phase)  #reconstruct with new phase (elementwise mult)
            magphase[0]=0; magphase[numBins-1] = 0 #remove dc and nyquist
            m_recon=np.concatenate([magphase,np.flip(np.conjugate(magphase[1:numBins-1]), 0)]) 
            
            #overlap and add
            m_recon=np.real(np.fft.ifft(m_recon))*m_win
            y_out[i*hop_length:i*hop_length+fftsize]+=m_recon
            
    return y_out


def magspect2audio(msgram, fftsize, hop_length)  :
    return spsi(msgram, fftsize, hop_length)

def logspect2audio(lsgram, fftsize, hop_length) :
    return spsi(np.power(10, lsgram/20), fftsize, hop_length)






