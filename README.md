GLUING CODE INPUT	
1.	Raw netcdfs from PollyXTs
    If the code will be implemented before deadtime, background and range corrections:
      a. Dead time
      b. First bin
      c. First and last bin to calculate the background. For example, for AKY it is idx_min = first_bin // 10
          idx_max = first_bin * 9 // 10.
      d. Bin length to calculate the altitude for the range correction.

If the code will be implemented after range correction: 
2.	The interval for averaging over time
3.	The depol_cal_angle when it is in the normal position. For example, for AKY it is 0.0 degrees. (To avoid these profiles in averaging)
4.	The overlap (in bins)
5.	The sensitivity parameters: correlation coefficient threshold and signal to noise ratio threshold
