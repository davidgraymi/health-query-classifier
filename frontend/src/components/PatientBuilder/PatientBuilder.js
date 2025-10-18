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

import React, { useState, useEffect } from "react";
import "./PatientBuilder.css";
import { JsonViewer } from "@textea/json-viewer"; // updated import
import DetailsPopup from "../DetailsPopup/DetailsPopup";

// Global caching function to load patients & conditions once
let cachedPatientsAndConditions = null;
function getPatientsAndConditions() {
  if (cachedPatientsAndConditions)
    return Promise.resolve(cachedPatientsAndConditions);
  return fetch("/assets/patients_and_conditions.json")
    .then((response) => response.json())
    .then((data) => {
      cachedPatientsAndConditions = data;
      return data;
    });
}

const PatientBuilder = ({
  selectedPatient,
  selectedCondition,
  setSelectedPatient,
  setSelectedCondition,
  onNext,
  onBack,
}) => {
  const [patients, setPatients] = useState([]);
  const [conditions, setConditions] = useState([]);
  const [hoveredPatient, setHoveredPatient] = useState(null);
  const [isVideoLoading, setIsVideoLoading] = useState(false);

  const [isPopupOpen, setIsPopupOpen] = useState(false);
  const [popupJson, setPopupJson] = useState(null);
  const [isDetailsPopupOpen, setIsDetailsPopupOpen] = useState(false);


  useEffect(() => {
    getPatientsAndConditions()
      .then((data) => {
        setPatients(data.patients);
        setConditions(data.conditions);
      })
      .catch((error) =>
        console.error("Error fetching patients and conditions:", error)
      );
  }, []);

  useEffect(() => {
    if (
      selectedPatient &&
      selectedPatient.existing_condition !== "depression" &&
      selectedCondition === "Serotonin Syndrome"
    ) {
      setSelectedCondition(null);
    }
  }, [selectedPatient]);

  // When a new patient is selected, set the video to a loading state
  // to ensure the placeholder image is shown.
  useEffect(() => {
    if (selectedPatient) {
      setIsVideoLoading(true);
    }
  }, [selectedPatient]);

  const handleGo = () => {
    if (selectedPatient && selectedCondition) {
      onNext();
    }
  };

  const openPopup = (patient) => {
    if (patient && patient.fhirFile) {
      fetch(patient.fhirFile)
        .then((response) => response.json())
        .then((json) => {
          setPopupJson(json);
          setIsPopupOpen(true);
        })
        .catch((error) => console.error("Error fetching FHIR JSON:", error));
    }
  };

  const closePopup = () => {
    setIsPopupOpen(false);
    setPopupJson(null);
  };

  return (
    <div className="patient-builder-container">
      <div className="headerButtonsContainer">
      <button className="back-button" onClick={onBack}>
          <i className="material-icons back-button-icon">keyboard_arrow_left</i>
          Back
      </button>
      <button className="details-button" onClick={() => setIsDetailsPopupOpen(true)}>
          <i className="material-icons code-block-icon">code</i>&nbsp;
          Details about this Demo
      </button>
      </div>
      <div className="frame">
        <div className="selection-section">
          <div className="header2">Select a Patient</div>
          <div className="patient-list">
            {patients.map((patient) => {
              const isSelected = selectedPatient && selectedPatient.id === patient.id;
              return (
                <div
                  key={patient.id}
                  className="patient-card"
                >
                  <div
                    className={`patient-video-container ${isSelected ? "selected" : ""}`}
                    onClick={() => setSelectedPatient(patient)}
                  >
                    <img
                      src={patient.img}
                      className="patient-img"
                      alt={patient.name}
                      draggable="false"
                      onDragStart={(e) => e.preventDefault()}
                      style={{ opacity: isSelected && !isVideoLoading ? 0 : 1 }}
                    />
                    {isSelected && (
                      <video
                        key={patient.id}
                        src={patient.video}
                        className="patient-video"
                        autoPlay
                        muted
                        loop
                        onCanPlay={() => setIsVideoLoading(false)}
                        style={{ opacity: isVideoLoading ? 0 : 1 }}
                      />
                    )}
                    <div className="ehr-label" onClick={(e) => { e.stopPropagation(); openPopup(patient); }}>
                      Synthetic Health Record (FHIR)
                    </div>
                  </div>
                  <div className="patient-info">
                    <div className="category-value">
                      {patient.name}, {patient.age} years old, {patient.gender}
                    </div>
                    <div className="category-value">
                      Existing condition: {patient.existing_condition}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        <div className="selection-section">
          <div className="header2">Explore a Condition</div>
          <div className="lighttext">
          In this demonstration, a persona, simulated using Gemini 2.5 Flash, will interact with an AI agent, built with MedGemma. 
          Neither the simulated persona nor the AI agent have been provided the diagnosis for the current condition (selected below). 
          The AI agent facilitates structured information-gathering, designed to usefully collect and summarize the patient's symptoms. 
          For the purposes of this demonstration, the AI agent also has access to elements of the patient's health record (provided as FHIR resources).
          </div>
          <div className="condition-list">
            {conditions.map((cond) => {
              const isDisabled =
                cond.name === "Serotonin Syndrome" &&
                selectedPatient &&
                selectedPatient.existing_condition !== "Depression";
              return (
                <div
                  key={cond.name}
                  className={`condition-card lighttext ${
                    selectedCondition === cond.name ? "selected" : ""
                  } ${isDisabled ? "disabled" : ""}`}
                  onClick={
                    !isDisabled
                      ? () => setSelectedCondition(cond.name)
                      : undefined
                  }
                >
                  <div><strong>{cond.name}</strong></div>
                  <div>{cond.description}</div>
                </div>
              );
            })}
          </div>
        </div>
        <button
          className="info-button"
          onClick={handleGo}
          disabled={!(selectedPatient && selectedCondition)}
        >
          Launch simulation
        </button>
      </div>
      {isPopupOpen && (
        <div className="popup-overlay" onClick={closePopup}>
          <div
            className="popup-content json-popup-content"
            onClick={(e) => e.stopPropagation()}
          >
            <h2>Synthetic Electronic Health Record</h2>
            <span>This is a sample of the patient’s electronic health record, shown in a standard (FHIR) format. This FHIR record, like the patient, was generated solely for the purposes of this demo.</span>
            <div className="json-viewer-container">
              <JsonViewer value={popupJson} theme="monokai" />
            </div>
            <button className="popup-button" onClick={closePopup}>
              Close
            </button>
          </div>
        </div>
      )}
       <DetailsPopup
        isOpen={isDetailsPopupOpen}
        onClose={() => setIsDetailsPopupOpen(false)}
      />
    </div>
  );
};

export default PatientBuilder;
