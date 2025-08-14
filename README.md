Here’s the extracted text from your PDF:

---

**phase1**

**IISc Aerospace Engineering Internship**
**LangChain + Django Creative Assignment**

**Important Dates and Submission**

* **Submission Deadline:** 14th August 2025, 11:59 PM IST (Late submissions will not be accepted)
* **Submission Link:** Submit your assignment once only via the designated Google form:
  [https://docs.google.com/forms/d/e/1FAIpQLSfzzxt\_WNCUvX8tpveFSWUCsYRjlW03S4-jBM2HUOc1nGLUkA/viewform?usp=header](https://docs.google.com/forms/d/e/1FAIpQLSfzzxt_WNCUvX8tpveFSWUCsYRjlW03S4-jBM2HUOc1nGLUkA/viewform?usp=header)

**Internship Details**

* This internship is unpaid.
* **Work Mode:** On-site preferred but optional; hybrid or remote options may be discussed
* **Selection Process:** Qualified candidates will be invited for technical interviews, where you will explain your code and design decisions

**Objective**
Build a Django web application powered by LangChain that:

1. Takes a user prompt as text input (optional audio → transcription allowed)
2. Generates:

   * Short story
   * Detailed character description
   * Detailed background/scene description
3. Uses the character description to build a prompt for character image generation
4. Uses the background description to build a prompt for background image generation
5. Generates two separate images (character + background) using free/open-source AI models
6. Combines these into a unified scene image
7. Displays story text, character description, and combined image on the web interface

**Key Requirements**

**Web Interface (Django)**

* Input form: text prompt or optional audio upload
* Output:

  * Short story
  * Character description
  * Combined scene image
* Flexible frontend: Django templates, React, Vue, etc., integrated with Django backend

**LangChain Orchestration**

* Separate chains/agents for:

  1. Story + descriptions generation (short story, detailed character description, detailed background description)
  2. Character image generation (prompt from character description)
  3. Background image generation (prompt from background description)
* Character & background descriptions must be used to craft exact image prompts — don’t feed raw user prompt to image models
* Modular design with clear prompt templates and iterative refinement
* Robust error handling and logging

**Image Generation**

* Only use free or open-source tools (Stable Diffusion, Flux, AI Studio, Hugging Face models)
* No paid/subscription APIs (e.g., Runway)
* Ensure visual style and perspective match between character & background
* Merge images into one coherent scene using Python tools (PIL, OpenCV)

**Deliverables**

1. Complete Django project source code with LangChain orchestration
2. `requirements.txt` listing dependencies
3. README with setup instructions and architecture overview
4. Prompt engineering documentation (.md or .pdf) detailing:

   * How the story prompt works
   * How character & background prompts are constructed
   * Example outputs and iterations
5. Optional: Sample generated outputs (images + text)

**Evaluation Criteria**

* Prompt engineering creativity & clarity
* Completeness & elegance of LangChain orchestration
* Narrative quality & coherence
* Visual appeal & integration of images
* Usability & design of the web interface
* Robustness (error handling, logging), modularity
* Documentation quality
* Compliance with free/open AI tools policy

**Additional Instructions**

* Submit once via provided Google form
* Keep backups of all work
* Be ready to explain code & design choices in interview

**Contact for Queries**
📧 [bsvivek2003@gmail.com](mailto:bsvivek2003@gmail.com)
