# DNA, RNA, and Protein Classification Project

This project focuses on classifying DNA, RNA, and protein sequences using a combination of machine learning and deep learning techniques. It includes a backend server, a Vue.js-based frontend, and a Jupyter Notebook for model training and testing.

---

## Project Structure

```
project-directory/
├── frontend/       # Frontend application built with Vue.js
├── backend/        # Backend server implemented in Python
├── models.ipynb    # Jupyter Notebook for model training and evaluation
```

---

## Getting Started

### Backend Setup

1. Navigate to the `backend` folder:
   ```bash
   cd backend
   ```

2. Run the backend server using Python:
   ```bash
   python app.py
   ```

   The backend will start listening for requests at `http://127.0.0.1:5001` (default Flask host and port).

---

### Frontend Setup

1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```

2. Ensure you are using Node.js version 16 (using nvm):
   ```bash
   nvm use 16
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend will run locally, typically accessible at `http://localhost:8013` (or another port as indicated).

   The frontend is styled using the open-source template [smpe-admin-web](https://github.com/sanyueruanjian/smpe-admin-web.git) for an enhanced user experience.

---

### Using the Application

Once both the backend and frontend services are running:

1. Open the frontend application in your web browser.
2. Input biological sequence details such as DNA, RNA, or protein data.
3. Submit the data to classify the macromolecule type using the models deployed on the backend.

---

## Model Training and Testing

The `models.ipynb` file contains the full process of training, optimizing, and evaluating machine learning and deep learning models for sequence classification. It includes:

- Data preprocessing steps.
- Model selection and hyperparameter tuning.
- Performance evaluation with test results.

To view and run the notebook:

1. Open the file in Jupyter Notebook or any compatible IDE (e.g., JupyterLab, VSCode).
2. Execute the cells to reproduce the training and testing process.

---

## Dependencies

### Backend:
- Python 3.x
- Flask
- scikit-learn
- PyTorch

### Frontend:
- Node.js (version 16)
- Vue.js
- Dependencies listed in `package.json`

---

## Notes

- Ensure that the backend server is running before using the frontend to ensure proper functionality.
- Model training is computationally intensive; it is recommended to use a machine with GPU acceleration for faster training times.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

Special thanks to the creators of the [smpe-admin-web](https://github.com/sanyueruanjian/smpe-admin-web.git) template for their excellent frontend design.
