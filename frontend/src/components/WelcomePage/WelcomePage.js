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

import React from 'react';
import './WelcomePage.css';

const WelcomePage = ({ onSwitchPage }) => {
  return (
    <div className="welcome page">
      <img src="/assets/medgemma.avif" alt="MedGemma Logo" className="medgemma-logo" />
      <div className="info-page-container">
        <div className="graphics">
          <img className="graphics-top" src="/assets/welcome_top_graphics.svg" alt="Welcome top graphics" />
          <img className="graphics-bottom" src="/assets/welcome_bottom_graphics.svg" alt="Welcome bottom graphics" />
        </div>
        <div className="info-content">
          <div className="info-header">
            <span className="title-header">Simulated Pre-visit Intake Demo</span>
          </div>
          <div className="info-text">
            Healthcare providers often need to gather patient information before appointments. 
            This demo illustrates how MedGemma could be used in an application to streamline pre-visit information collection and utilization. 
            <br /><br/>
            First, a pre-visit AI agent built with MedGemma asks questions to gather information.
            After it has identified and collected relevant information, the demo application generates a pre-visit report. 
            <br /><br/>
            This type of intelligent pre-visit report can help providers be more efficient and effective while also providing an improved experience 
            for patients relative to traditional intake forms.
            <br /><br/>
            Lastly, you can view an evaluation of the pre-visit report which provides insights into the quality of the output. 
            For this evaluation, MedGemma is provided the reference diagnosis, allowing "self-evaluation" that highlights both strengths and what it could have done better.
          </div>
          <div className="info-disclaimer-text">
            <span className="info-disclaimer-title">Disclaimer</span> This
            demonstration is for illustrative purposes only and does not represent a finished or approved
            product. It is not representative of compliance to any regulations or standards for
            quality, safety or efficacy. Any real-world application would require additional development,
            training, and adaptation. The experience highlighted in this demo shows MedGemma's baseline
            capability for the displayed task and is intended to help developers and users explore possible
            applications and inspire further development.
          </div>
          <button className="info-button" onClick={onSwitchPage}>Select Patient</button>
        </div>
      </div>
    </div>
  );
};

export default WelcomePage;
