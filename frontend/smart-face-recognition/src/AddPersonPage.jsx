import React, { useState } from "react";

function AddPersonPage() {
  const [name, setName] = useState("");
  const [images, setImages] = useState([]); // <-- changed from single image

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!name || images.length === 0) {
      alert("Please enter a name and choose at least one image.");
      return;
    }

    const formData = new FormData();
    formData.append("name", name);
    for (let i = 0; i < images.length; i++) {
      formData.append("files", images[i]); // matches FastAPI's List[UploadFile] param
    }

    try {
      const response = await fetch("http://127.0.0.1:8000/upload-images", { // <-- update port & endpoint
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        alert(`Person added successfully! ${data.files.length} images uploaded.`);
        setName("");
        setImages([]);
      } else {
        alert(data.message || "Upload failed!");
      }
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend.");
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      <div className="text-center mb-12">
        <h2 className="text-4xl sm:text-5xl font-extrabold mb-4">Add New Person</h2>
        <p className="text-lg text-gray-300 max-w-3xl mx-auto">
          Upload clear face images to add a person to the Smart Face Recognition database.
        </p>
      </div>

      <div className="bg-slate-800 rounded-lg shadow-lg p-6 max-w-2xl mx-auto">
        <h3 className="text-2xl font-semibold mb-6">Person Details</h3>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-lg font-medium text-gray-300">Full Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="mt-1 w-full bg-gray-700 text-white rounded-md py-2 px-4 focus:ring-2 focus:ring-blue-600"
              placeholder="Enter full name"
            />
          </div>

          <div>
            <label className="block text-lg font-medium text-gray-300">Upload Face Images</label>
            <input
              type="file"
              accept="image/*"
              multiple // <-- allow multiple selection
              onChange={(e) => setImages(Array.from(e.target.files))}
              className="mt-2 w-full text-gray-300"
            />
          </div>

          <button
            type="submit"
            className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-md shadow-md transition-colors text-lg"
          >
            Add Person
          </button>
        </form>
      </div>
    </div>
  );
}

export default AddPersonPage;
