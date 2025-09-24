#!/bin/bash

# Script to set up GitHub remote for the enhanced scikit-learn fork
# Run this after creating the repository on GitHub

GITHUB_REPO="https://github.com/LakshmiSravya123/Scikit-learn-enhanced.git"

echo "Setting up GitHub remote..."
git remote add origin $GITHUB_REPO

echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo "GitHub repository setup complete!"
echo "Repository URL: $GITHUB_REPO"
