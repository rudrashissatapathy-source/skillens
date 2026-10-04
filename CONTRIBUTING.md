# Contributing to SkillLens 🎓

Thank you for your interest in contributing to **SkillLens**! We welcome community contributions, bug reports, feature suggestions, and pull requests to help advance placement readiness intelligence and explainable AI for students.

---

## 🛠️ Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/rudrashissatapathy-source/skillens.git
   cd skillens
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Tests:**
   ```bash
   pytest tests/test_pipeline.py -v
   ```

5. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Testing Guidelines

- All test cases live in `tests/test_pipeline.py`.
- Ensure new features have accompanying unit tests.
- Verify monotonicity guarantees if modifying `PlacementReadinessClassifier` in `src/model.py`.
- Verify SHAP attribution fallbacks in `src/explainability.py`.

---

## 📬 Pull Request Process

1. Create a descriptive branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Commit with concise, descriptive commit messages.
3. Ensure all tests pass: `pytest tests/test_pipeline.py`.
4. Push your branch and open a Pull Request against the `main` branch.

---

## 📜 Code of Conduct

Please maintain a collaborative, welcoming, and inclusive environment for all contributors.
