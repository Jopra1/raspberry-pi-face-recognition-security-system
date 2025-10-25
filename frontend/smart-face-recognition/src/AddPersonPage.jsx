import React from 'react';

function AddPersonPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h2 className="text-4xl sm:text-5xl font-extrabold mb-4">Add New Person</h2>
        <p className="text-lg text-gray-300 max-w-3xl mx-auto">
          Register a new individual in the Smart Face Recognition database. Provide details and capture a face image for accurate detection.
        </p>
      </div>

      {/* Form Card */}
      <div className="bg-slate-800 rounded-lg shadow-lg p-6 max-w-2xl mx-auto">
        <h3 className="text-2xl font-semibold mb-6">Person Details</h3>
        <form className="space-y-6">
          <div>
            <label htmlFor="name" className="block text-lg font-medium text-gray-300">
              Full Name
            </label>
            <input
              type="text"
              id="name"
              className="mt-1 w-full bg-gray-700 text-white rounded-md py-2 px-4 focus:outline-none focus:ring-2 focus:ring-blue-600"
              placeholder="Enter full name"
            />
          </div>
         
          <div>
            <label className="block text-lg font-medium text-gray-300">
              Capture Face
            </label>
            <button
              type="button"
              className="mt-1 w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-md shadow-md transition-colors text-lg"
            >
              Capture Image
            </button>
          </div>
          <button
            type="submit"
            className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-md shadow-md transition-colors text-lg"
          >
            Add Person
          </button>
        </form>
        <p className="text-gray-400 mt-6">
          Note: Ensure the camera is ready to capture a clear face image for the database.
        </p>
      </div>

      {/* Additional Info */}
      <div className="mt-12 text-center">
        <h3 className="text-2xl font-semibold mb-4">About the Database</h3>
        <p className="text-gray-400 max-w-3xl mx-auto">
          The Smart Face Recognition system stores face encodings securely on the Raspberry Pi. Added persons are instantly available for detection.
        </p>
      </div>
    </div>
  );
}

export default AddPersonPage;