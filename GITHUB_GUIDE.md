# GitHub setup guide

## 1. Create a repository on GitHub

1. Sign in at https://github.com.
2. Click **New repository**.
3. Use a name such as `custom-logistic-regression-churn`.
4. Add a short description.
5. Choose **Private** while developing if you prefer to keep college work private; you can change visibility later.
6. Do not initialize the GitHub repository with a README if you are going to push this existing project folder (this ZIP already includes one).

## 2. Open a terminal in the extracted project folder

Check that you are in the directory containing `README.md`.

```bash
git --version
```

If Git is not installed, install it from https://git-scm.com/downloads.

## 3. Initialize Git and push

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git init
git add .
git status
git commit -m "Initial project structure"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/custom-logistic-regression-churn.git
git push -u origin main
```

If Git asks you to authenticate, follow GitHub's browser/device login flow or configure a supported credential manager. Do not put a password or access token directly into source files.

## 4. Check the repository

Refresh your GitHub repository page. Confirm that README.md, requirements.txt, app.py, src/, notebooks/, tests/, and report/ are visible.

## 5. Everyday workflow

After changing code:

```bash
git status
git add src app.py tests
git commit -m "Implement model evaluation improvements"
git push
```

Use a meaningful commit message that explains the change.

## 6. Important: do not upload generated or private files

The included `.gitignore` excludes the raw CSV, virtual environment, generated model bundle, and output files. Review `git status` before every commit. Never commit credentials, access tokens, private customer data, or files you do not have permission to share.

## 7. Optional: GitHub web upload

For a very small change, GitHub's web interface can upload files. For a complete project with multiple folders and future development, Git is recommended. Official guide: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
