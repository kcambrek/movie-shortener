import { useState, useRef } from 'react';
import Checkbox from '@mui/material/Checkbox';
import FormControlLabel from '@mui/material/FormControlLabel';
import Grid from '@mui/material/Grid';
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControl from '@mui/material/FormControl';
import FormLabel from '@mui/material/FormLabel';

// ========== UI Components ==========

function Slider({ description, min, max, value, setValue, step = 0.1 }) {
  return (
    <div>
      <label>{`${description} (${value})`}</label>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        step={step}
        onChange={(e) => setValue(e.target.value)}
      />
    </div>
  );
}

function VideoUpload({ setVideoMetadata }) {
  function handleVideoUpload(uploadedFile) {  
    const videoUrl = URL.createObjectURL(uploadedFile);
    setVideoMetadata(prev => ({...prev, url: videoUrl}));
  }

  return (
    <div>
      <input 
        type="file" 
        accept="video/*" 
        onChange={(e) => handleVideoUpload(e.target.files[0])} 
      />
    </div>
  );
}

function VideoPlayer({ videoMetadata, setVideoMetadata, videoRef }) {
  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setVideoMetadata(prev => ({
        ...prev,
        duration: videoRef.current.duration,
        width: videoRef.current.videoWidth,
        height: videoRef.current.videoHeight,
        currentTime: 0
      }));
    }
  };

  return (
    <div style={{ width: '100%', position: 'relative', paddingTop: '56.25%' }}>
      <video 
        ref={videoRef}
        src={videoMetadata.url} 
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          objectFit: 'contain'
        }}
        onLoadedMetadata={handleLoadedMetadata}
        controls
      />
    </div>
  );
}

function FrameImage({ selected, setSelected, frameImage, time }) {
  return (
    <div className="frame-container">
      <img 
        src={frameImage} 
        alt={`Frame at ${time}s`}
        style={{ width: '160px', height: 'auto' }}
      />
      <div className="time-label">{time.toFixed(2)}s</div>
      <FormControlLabel
        control={
          <Checkbox 
            checked={selected}
            onChange={setSelected}
            color="primary"
          />
        }
        label={selected ? "Selected" : "Select"}
      />
    </div>
  );
}

function WhatToDetectButton({ setWhatToDetect }) {
  return (
    <FormControl>
      <FormLabel id="demo-radio-buttons-group-label">Looking for</FormLabel>
      <RadioGroup
        aria-labelledby="demo-radio-buttons-group-label"
        defaultValue="scene-changes"
        name="radio-buttons-group"
        onChange={(e) => setWhatToDetect(e.target.value)}
      >
        <FormControlLabel value="scene-changes" control={<Radio />} label="Scene Changes" />
        <FormControlLabel value="continuous-action" control={<Radio />} label="Continuous Action" />
      </RadioGroup>
    </FormControl>
  );
}

// ========== Main App Component ==========

export default function App() {
  // State variables

  const [sampledFrames, setSampledFrames] = useState([]);
  const [imagesSelected, setImagesSelected] = useState(Array(sampledFrames.length).fill(false));
  const [sampleRate, setSampleRate] = useState(1);
  const [segmentDuration, setSegmentDuration] = useState(0.5);
  const [videoMetadata, setVideoMetadata] = useState({duration: 0, width: 0, height: 0, currentTime: 0, url: "", sceneChangePeakIndices: [], continuousActionPeakIndices: []});
  const [finishedSampling, setFinishedSampling] = useState(false);
  const [finishedFindingPeaks, setFinishedFindingPeaks] = useState(false);
  const [kernelSize, setKernelSize] = useState(5);
  const [distanceBetweenPeaks, setDistanceBetweenPeaks] = useState(9);
  const [numSegments, setNumSegments] = useState(60);
  const [whatToDetect, setWhatToDetect] = useState("scene-changes");
  const [showGrid, setShowGrid] = useState(false);
  const videoRef = useRef(null);
  

  // Event handlers
  const handleSampleRateChange = (val) => {
    const numVal = Number(val);
    setSampleRate(numVal);
    setSegmentDuration((prev) => Math.min(prev, numVal));
  };

  const handleSegmentDurationChange = (val) => {
    setSegmentDuration(Number(val));
  };

  const handleImageSelect = (index) => {
    const newImagesSelected = [...imagesSelected];
    newImagesSelected[index] = !newImagesSelected[index];
    setImagesSelected(newImagesSelected);
  };

  const handleKernelSizeChange = (val) => {
    setKernelSize(Number(val));
  };

  const handleDistanceBetweenPeaksChange = (val) => {
    setDistanceBetweenPeaks(Number(val));
  };

  const handleNumSegmentsChange = (val) => {
    setNumSegments(Number(val));
  };

  return (
    <div>
      {/* Video Upload and Player */}
      <VideoUpload setVideoMetadata={setVideoMetadata} /> 
      {videoMetadata.url && <VideoPlayer videoMetadata={videoMetadata} setVideoMetadata={setVideoMetadata} videoRef={videoRef} />}
      
      {/* Sampling Controls */}
      <Slider
        description="Frame Sampling Interval (seconds)"
        min={1}
        max={5}
        value={sampleRate}
        step={0.5}
        setValue={handleSampleRateChange}
      />
      <Slider
        description="Segment Duration (seconds)"
        min={0.5}
        max={sampleRate}
        value={segmentDuration}
        step={0.5}
        setValue={handleSegmentDurationChange}
      />
      <button onClick={() => {
        setFinishedSampling(false);
        setFinishedFindingPeaks(false);
        sample({
          videoRef, 
          videoMetadata, 
          sampleRate, 
          segmentDuration, 
          setSampledFrames, 
          setImagesSelected, 
          setFinishedSampling
        });
      }}>
        Sample
      </button>
      
      {/* Analysis Controls (shown after sampling) */}
      {finishedSampling && (
        <>
          <Slider
            description="Kernel Size"
            min={3}
            max={15}
            value={kernelSize}
            step={2}
            setValue={handleKernelSizeChange}
          />
          <Slider
            description="Distance between peaks"
            min={1}
            max={20}
            value={distanceBetweenPeaks}
            step={1}
            setValue={handleDistanceBetweenPeaksChange}
          />
          <WhatToDetectButton setWhatToDetect={setWhatToDetect} />
          <button onClick={() => {
            setFinishedFindingPeaks(false);
            findPeaks({
              hashDistances: sampledFrames.map(frame => frame.hashDistance),
              inverse: whatToDetect === "scene-changes",
              kernelSize,
              distanceBetweenPeaks,
              setVideoMetadata
            }); 
            setFinishedFindingPeaks(true);
          }}>
            Find Peaks
          </button>
        </>
      )}
      
      {/* Display sampled frames */}
      {finishedSampling && finishedFindingPeaks && (
        <>      
        <Grid container spacing={2}>
            {sampledFrames
              .filter((frame, i) => {
                // Use the appropriate peak indices based on whatToDetect
                const peakIndices = whatToDetect === "scene-changes" 
                  ? videoMetadata.sceneChangePeakIndices 
                  : videoMetadata.continuousActionPeakIndices;
                return peakIndices.includes(i);
              })
              .sort((a, b) => a.time - b.time)
              .map((frame, i) => (
                <Grid item xs={12} sm={6} md={4} key={i}>
                  <FrameImage
                    selected={imagesSelected[sampledFrames.indexOf(frame)]}
                    setSelected={() => handleImageSelect(sampledFrames.indexOf(frame))}
                    frameImage={frame.frameImage}
                    time={frame.time}
                  />
                </Grid>
              ))}
          </Grid>
        </>
      )}
    </div>
  );
}

// ========== Video Processing Functions ==========

async function sample({videoRef, videoMetadata, sampleRate, segmentDuration, setSampledFrames, setImagesSelected, setFinishedSampling}) {
  if (!videoRef.current) {
    console.error('No video element available');
    return;
  }

  videoRef.current.controls = false;
  videoRef.current.currentTime = 0;

  let previousHash;
  let processedFrames = 0;
  let dedupedFrames = 0;

  // Clear existing frames
  setSampledFrames([]);
  setImagesSelected([]);

  // Reset video to beginning
  for (let i = 0; i < videoMetadata.duration; i = i + parseFloat(sampleRate)) {
    videoRef.current.currentTime = i;
    await new Promise(resolve => {
      videoRef.current.onseeked = resolve;
    });
    
    const frameCanvas = document.createElement('canvas');
    frameCanvas.width = videoMetadata.width;
    frameCanvas.height = videoMetadata.height;
    const frameCtx = frameCanvas.getContext('2d');
    frameCtx.drawImage(videoRef.current, 0, 0, frameCanvas.width, frameCanvas.height);
    const frameDataURL = frameCanvas.toDataURL('image/jpeg', 1);

    const currentHash = await computeImageHash(frameDataURL);
    let hashDistance;

    if (i === 0) {
      hashDistance = 10;
    } else {
      hashDistance = hammingDistance(previousHash, currentHash);
      console.log(`Frame at ${i}s, hash distance: ${hashDistance}`);
    }

    previousHash = currentHash;

    if (hashDistance > 5) {
      const frameData = {
        time: i,
        frameImage: frameDataURL,
        selected: true,
        hashDistance: hashDistance
      };
      
      // Update state with the new frame immediately
      setSampledFrames(prev => {
        const newFrames = [...prev, frameData];
        // Also update imagesSelected to match the new length
        setImagesSelected(Array(newFrames.length).fill(false));
        return newFrames;
      });
    } else {
      dedupedFrames++;
    }
    processedFrames++;
  }

  console.log(`Finished processing ${processedFrames} frames`);
  console.log(`Deduplicated ${dedupedFrames} frames`);

  videoRef.current.controls = true;
  videoRef.current.currentTime = 0;
  setFinishedSampling(true);
}

// ========== Utility Functions ==========

function computeImageHash(dataURL) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const size = 32;
      const lowSize = 8;

      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, size, size);

      const imageData = ctx.getImageData(0, 0, size, size).data;

      // Step 1: Get grayscale pixels
      const gray = [];
      for (let i = 0; i < imageData.length; i += 4) {
        const r = imageData[i];
        const g = imageData[i + 1];
        const b = imageData[i + 2];
        gray.push((r + g + b) / 3);
      }

      // Convert flat array to 2D
      const pixels = [];
      for (let y = 0; y < size; y++) {
        pixels.push(gray.slice(y * size, (y + 1) * size));
      }

      // Step 2: Apply 2D DCT
      const dct = dct2D(pixels);

      // Step 3: Take top-left 8x8 DCT block
      const topLeft = [];
      for (let i = 0; i < lowSize; i++) {
        for (let j = 0; j < lowSize; j++) {
          topLeft.push(dct[i][j]);
        }
      }

      // Step 4: Median threshold
      const median = topLeft.slice().sort((a, b) => a - b)[Math.floor(topLeft.length / 2)];
      const hashBits = topLeft.map(v => v > median ? '1' : '0');

      resolve(hashBits.join(''));
    };
    img.src = dataURL;
  });
}

function dct2D(matrix) {
  const N = matrix.length;
  const result = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let u = 0; u < N; u++) {
    for (let v = 0; v < N; v++) {
      let sum = 0;
      for (let x = 0; x < N; x++) {
        for (let y = 0; y < N; y++) {
          sum += matrix[x][y] *
            Math.cos(((2 * x + 1) * u * Math.PI) / (2 * N)) *
            Math.cos(((2 * y + 1) * v * Math.PI) / (2 * N));
        }
      }
      const c_u = u === 0 ? 1 / Math.sqrt(2) : 1;
      const c_v = v === 0 ? 1 / Math.sqrt(2) : 1;
      result[u][v] = 0.25 * c_u * c_v * sum;
    }
  }
  return result;
}

function hammingDistance(hash1, hash2) {
  if (hash1.length !== hash2.length) {
    throw new Error('Hash lengths must match');
  }
  let distance = 0;
  for (let i = 0; i < hash1.length; i++) {
    if (hash1[i] !== hash2[i]) {
      distance++;
    }
  }
  return distance;
}

function medianFilter(signal, kernelSize=5) {
  const half = Math.floor(kernelSize / 2);
  const padded = Array(half).fill(signal[0]).concat(signal, Array(half).fill(signal[signal.length - 1]));
  const result = [];

  for (let i = 0; i < signal.length; i++) {
    const window = padded.slice(i, i + kernelSize);
    const sorted = window.slice().sort((a, b) => a - b);
    result.push(sorted[Math.floor(kernelSize / 2)]);
  }

  return result;
}

function findPeaks({hashDistances, inverse, kernelSize, distanceBetweenPeaks, prominence = 1, setVideoMetadata}) {
  console.log("finding peaks, inverse: ", inverse);
  let signal = inverse ? hashDistances.map(x => -x) : hashDistances;
  
  // Apply median filter (already implemented correctly)
  const filteredSignal = medianFilter(signal, kernelSize);
  
  // Find all potential peaks (points higher than both neighbors)
  const peakIndices = [];
  for (let i = 1; i < filteredSignal.length - 1; i++) {
    if (filteredSignal[i] > filteredSignal[i - 1] && filteredSignal[i] > filteredSignal[i + 1]) {
      peakIndices.push(i);
    }
  }
  
  // Calculate prominence for each peak and filter by prominence
  const prominentPeaks = [];
  for (const peakIdx of peakIndices) {
    const peakHeight = filteredSignal[peakIdx];
    
    // Find left and right bases (lowest points on either side until higher peak)
    let leftBase = peakHeight;
    for (let i = peakIdx - 1; i >= 0; i--) {
      if (filteredSignal[i] > peakHeight) break;
      leftBase = Math.min(leftBase, filteredSignal[i]);
    }
    
    let rightBase = peakHeight;
    for (let i = peakIdx + 1; i < filteredSignal.length; i++) {
      if (filteredSignal[i] > peakHeight) break;
      rightBase = Math.min(rightBase, filteredSignal[i]);
    }
    
    const peakProminence = peakHeight - Math.max(leftBase, rightBase);
    
    if (peakProminence >= prominence) {
      prominentPeaks.push({index: peakIdx, prominence: peakProminence});
    }
  }
  
  // Sort by prominence in descending order
  prominentPeaks.sort((a, b) => b.prominence - a.prominence);
  
  // Apply distance filter (keeping highest prominence peaks)
  const finalPeaks = [];
  const isExcluded = Array(filteredSignal.length).fill(false);
  
  for (const peak of prominentPeaks) {
    if (!isExcluded[peak.index]) {
      finalPeaks.push(peak.index);
      
      // Mark nearby points as excluded
      for (let i = Math.max(0, peak.index - distanceBetweenPeaks); 
           i <= Math.min(filteredSignal.length - 1, peak.index + distanceBetweenPeaks); 
           i++) {
        isExcluded[i] = true;
      }
      // Keep the current peak available
      isExcluded[peak.index] = false;
    }
  }
  
  // Sort by index for chronological order
  finalPeaks.sort((a, b) => a - b);
  
  if (inverse) {
    setVideoMetadata(prev => ({...prev, sceneChangePeakIndices: finalPeaks}));
  } else {
    setVideoMetadata(prev => ({...prev, continuousActionPeakIndices: finalPeaks}));
  }
  console.log("final peaks: ", finalPeaks);
  return finalPeaks;
}

