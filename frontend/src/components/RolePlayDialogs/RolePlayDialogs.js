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

import React, { useState } from "react";
import "./RolePlayDialogs.css";
import DetailsPopup from "../DetailsPopup/DetailsPopup";

const RolePlayDialogs = ({
  selectedPatient,
  selectedCondition,
  onStart,
  onBack,
}) => {
  const [isDetailsPopupOpen, setIsDetailsPopupOpen] = useState(false);

  return (
    <div className="page">
      <div className="headerButtonsContainer">
        <button className="back-button" onClick={onBack}>
          <i className="material-icons back-button-icon">keyboard_arrow_left</i>
          Back
        </button>
        <button className="details-button" onClick={() => setIsDetailsPopupOpen(true)}>
          <i className="material-icons code-block-icon">code</i>&nbsp; Details
          about this Demo
        </button>
      </div>
      <div className="frame role-play-container">
        <div className="title-header">What’s happening in this simulation</div>
        <div className="dialogs-container">
          <div className="dialog-box">
            <div className="dialog-title-text">Pre-visit AI agent</div>
            <div className="dialog-subtitle">
              Built with: <img src="assets/medgemma.avif" height="16px" />{" "}
              27b
            </div>
            <img
              src="assets/ai_headshot.svg"
              alt="AI Avatar"
              className="ai-avatar"
            />
            <div className="dialog-body-scrollable">
              In this demo, MedGemma functions as an AI agent designed to assist in pre-visit information
              collection. It will interact with the patient agent to gather relevant data.
              To provide additional context, MedGemma also has access to information from the patient's EHR (in FHIR format).
              However, MedGemma is not provided the specific diagnois ({selectedCondition}).
              MedGemma's goal is to gather details about symptoms, relevant history,
              and current concerns to generate a comprehensive pre-visit report. 
            </div>
          </div>
          <div className="dialog-box">
            <div className="dialog-title-text">
              Patient persona: {selectedPatient.name}
            </div>
            <div className="dialog-subtitle">
              Simulated by:{" "}Gemini 2.5 Flash
            </div>
            <img
              src={selectedPatient.headshot}
              alt="Patient Avatar"
              className="patient-avatar"
            />
            <div className="dialog-body-scrollable">
              Gemini is provided a persona and information to play the role of the patient, {selectedPatient.name}.
              In this simulation, the patient agent does not know their diagnosis,
              but is experiencing related symptoms and concerns that can be shared during the interview. 
              To simulate a real-world situation with confounding information, additional information unrelated to the presenting condition has also been provided. 
            </div>
          </div>
        </div>
        <div className="report-notice">
          As the conversation develops, MedGemma <span className="highlight">creates and continually updates
          a real-time pre-visit report</span> capturing relevant
          information. Following pre-visit report generation, an evaluation is available. The purpose of this evaluation is to provide the viewer insights into quality of the output.
          For this evaluation, MedGemma is provided the previously unknown reference diagnosis, and is prompted to generate a 
          <span className="highlight">self evaluation that highlights strengths as well opportunities where the conversation and report could have been improved.</span>
        </div>
        <button className="info-button" onClick={onStart}>
          Start conversation
        </button>
      </div>
      <DetailsPopup
        isOpen={isDetailsPopupOpen}
        onClose={() => setIsDetailsPopupOpen(false)}
      />
    </div>
  );
};

export default RolePlayDialogs;
