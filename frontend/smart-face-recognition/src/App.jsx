import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './Navbar';
import HomePage from './HomePage';
import AddPersonPage from './AddPersonPage';
import './index.css'; // Assuming Tailwind CSS file

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-900 text-white">
        <Navbar />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/add-person" element={<AddPersonPage />} />
        </Routes>
     
      </div>
    </Router>
  );
}

export default App;