import React from 'react';
import { Link } from 'react-router-dom';

function Navbar() {
  return (
    <nav className="bg-slate-800 shadow-lg py-2 sticky top-0">
      <div className="w-full flex items-center justify-between px-4"> {/* full width */}
        {/* Logo/Title - Extreme Left */}
        <div className="flex-shrink-0">
          <h1 className="text-1xl sm:text-2xl font-bold tracking-tight">
            Smart Face Recognition
          </h1>
        </div>

        {/* Navigation Links - Extreme Right */}
        <div className="flex items-center gap-6">
          <Link
            to="/"
            className="px-5 py-2 rounded-md hover:bg-slate-700 transition-colors text-lg font-medium"
          >
            Detection
          </Link>
          <Link
            to="/add-person"
            className="px-5 py-2 rounded-md hover:bg-slate-700 transition-colors text-lg font-medium"
          >
            Add Person
          </Link>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
