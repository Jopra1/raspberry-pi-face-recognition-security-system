import React, { useState } from 'react';

function HomePage() {
  const [cameraStatus, setCameraStatus] = useState('Idle');
  const [error, setError] = useState(null);

  const startCamera = async () => {
    setCameraStatus('Starting...');
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/start-camera');
      const data = await response.json();
      if (response.ok) {
        setCameraStatus('Camera Running');
      } else {
        throw new Error(data.message || 'Failed to start camera');
      }
    } catch (err) {
      setCameraStatus('Idle');
      setError('Error starting camera: ' + err.message);
    }
  };

  const isRunning = cameraStatus === 'Camera Running';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Left: Big Video Display */}
        <div className="flex-1 bg-slate-800 rounded-lg shadow-lg p-4 lg:p-6">
          <h3 className="text-2xl font-semibold mb-4">Live Camera Feed</h3>
          <div className="bg-gray-900 flex-1 h-[600px] flex items-center justify-center rounded-md overflow-hidden">
            {isRunning ? (
              <img
                src="http://localhost:8000/video_feed"
                alt="Live Camera Feed"
                className="w-full h-full object-cover"
              />
            ) : (
              <p className="text-gray-400 text-lg">
                Camera feed will appear here.
              </p>
            )}
          </div>
        </div>

        {/* Right: Camera Control */}
        <div className="w-full lg:w-80 bg-slate-800 rounded-lg shadow-lg p-4 lg:p-6 flex flex-col justify-start">
          <h3 className="text-2xl font-semibold mb-4">Camera Control</h3>
          <button
            onClick={startCamera}
            disabled={cameraStatus === 'Starting...' || cameraStatus === 'Camera Running'}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold py-3 px-6 rounded-md shadow-md transition-colors text-lg"
          >
            {cameraStatus === 'Starting...' ? 'Starting...' : 'Start Camera'}
          </button>

          <div className="mt-4">
            {isRunning && <p className="text-green-400 text-lg">Camera is active!</p>}
            {error && <p className="text-red-400 text-lg">{error}</p>}
          </div>

          <div className="mt-6">
            <h4 className="text-lg font-medium text-gray-300 mb-2">System Status</h4>
            <p className="text-gray-400">
              {cameraStatus === 'Idle'
                ? 'Camera is idle. Click "Start Camera" to begin detection.'
                : cameraStatus === 'Starting...'
                ? 'Initializing camera...'
                : 'Camera is running and detecting faces.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HomePage;
