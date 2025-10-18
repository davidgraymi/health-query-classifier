/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React, { useState } from 'react';
import WelcomePage from './components/WelcomePage/WelcomePage';
import PatientBuilder from './components/PatientBuilder/PatientBuilder';
import RolePlayDialogs from './components/RolePlayDialogs/RolePlayDialogs';
import Interview from './components/Interview/Interview';
import PreloadImages from './components/PreloadImages';

const App = () => {
  const [currentPage, setCurrentPage] = useState('welcome');
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [selectedCondition, setSelectedCondition] = useState(null);

  const handleSwitchPage = () => {
    setCurrentPage('patientBuilder');
  };

  const handleSwitchToRolePlayDialogs = () => {
    setCurrentPage('rolePlayDialogs');
  };

  const handleSwitchToInterview = () => {
    setCurrentPage('interview');
  };

  const imageList = [
    '/assets/gemini.avif',
    '/assets/medgemma.avif',
    '/assets/ai_headshot.svg',
    '/assets/jordan_300.avif',
    '/assets/alex_300.avif',
    '/assets/sacha_150.avif',
    '/assets/jordan.avif',
    '/assets/alex.avif',
    '/assets/sacha.avif'
  ];

  return (
    <PreloadImages imageSources={imageList}>
      {currentPage === 'welcome' ? (
        <WelcomePage
          onSwitchPage={handleSwitchPage}
          setSelectedPatient={setSelectedPatient}
          setSelectedCondition={setSelectedCondition}
        />
      ) : currentPage === 'patientBuilder' ? (
        <PatientBuilder
          selectedPatient={selectedPatient}
          selectedCondition={selectedCondition}
          setSelectedPatient={setSelectedPatient}
          setSelectedCondition={setSelectedCondition}
          onNext={handleSwitchToRolePlayDialogs}
          onBack={() => setCurrentPage('welcome')} // Back to WelcomePage
        />
      ) : currentPage === 'rolePlayDialogs' ? (
        <RolePlayDialogs
          selectedPatient={selectedPatient}
          selectedCondition={selectedCondition}
          onStart={handleSwitchToInterview}
          onBack={() => setCurrentPage('patientBuilder')} // Back to PatientBuilder
        />
      ) : currentPage === 'interview' ? (
        <Interview
          selectedPatient={selectedPatient}
          selectedCondition={selectedCondition}
          onBack={() => setCurrentPage('rolePlayDialogs')} // Back to RolePlayDialogs
        />
      ) : null}
    </PreloadImages>
  );
};

export default App;
