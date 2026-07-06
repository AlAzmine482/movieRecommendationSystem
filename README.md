# 🎬 Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent movie recommendation system that suggests films based on genres, plot, and keywords using **Content-Based Filtering** with **TF-IDF** and **Cosine Similarity**. Built with Python and Streamlit for an interactive user experience.

## 🌟 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 📋 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Algorithm Details](#-algorithm-details)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## ✨ Features

- **Content-Based Filtering**: Recommends movies similar to a user's selected movie
- **Multi-Feature Analysis**: Considers genres, plot overview, and keywords
- **Interactive UI**: Built with Streamlit for a smooth user experience
- **Real-Time Recommendations**: Instantly generates recommendations with similarity scores
- **Rich Movie Details**: Displays ratings, release year, genres, and plot summaries
- **Visual Statistics**: Shows dataset insights and recommendation match percentages
- **Responsive Design**: Works on desktop and mobile devices

## 🎯 How It Works

```mermaid
graph LR
    A[User selects a movie] --> B[TF-IDF Vectorization]
    B --> C[Cosine Similarity Calculation]
    C --> D[Top 10 Similar Movies]
    D --> E[Display with ratings & details]
