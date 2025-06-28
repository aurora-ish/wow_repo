// Features.js - MuggleProof Luggage Control System

document.addEventListener("DOMContentLoaded", function () {
    // Initialize the app
    initializeApp();
  
    // Set up event listeners
    setupEventListeners();
  });
  
  // =================== INITIALIZATION ===================
  function initializeApp() {
    // Initialize all features
    updateGPSLocation();
    updateWeightDisplay();
    displayRandomQuote();
    setupHouseSelection();
  
    // Check for saved preferences
    checkSavedPreferences();
  
    console.log("MuggleProof Luggage Control System initialized");
  }
  
  function checkSavedPreferences() {
    // Check if user has previously selected a house theme
    const savedHouse = localStorage.getItem("selectedHouse");
    if (savedHouse) {
      applyHouseTheme(savedHouse);
      highlightSelectedHouse(savedHouse);
    }
  
    // Set toggle state from localStorage if available
    const followMeState = localStorage.getItem("followMeEnabled");
    if (followMeState === "true") {
      document.getElementById("followToggle").checked = true;
      updateFollowStatus(true);
    }
  }
  
  function setupEventListeners() {
    // GPS Location refresh button
    document
      .getElementById("refreshLocation")
      .addEventListener("click", updateGPSLocation);
  
    // Follow Me toggle
    document
      .getElementById("followToggle")
      .addEventListener("change", function (e) {
        updateFollowStatus(e.target.checked);
        localStorage.setItem("followMeEnabled", e.target.checked);
      });
  
    // House selection
    document.querySelectorAll(".house-option").forEach((option) => {
      option.addEventListener("click", function () {
        const house = this.getAttribute("data-house");
        applyHouseTheme(house);
        localStorage.setItem("selectedHouse", house);
        highlightSelectedHouse(house);
      });
    });
  
    // New quote button
    document
      .getElementById("newQuote")
      .addEventListener("click", displayRandomQuote);
  
    // Password update
    document
      .getElementById("updatePassword")
      .addEventListener("click", updatePassword);
  
    // Password reset
    document
      .getElementById("resetPassword")
      .addEventListener("click", resetPassword);
  
    // Sorting Hat
    document
      .getElementById("sortingHat")
      .addEventListener("click", consultSortingHat);
  }
  
  // =================== GPS LOCATION FEATURE ===================
  function updateGPSLocation() {
    const locationCoords = document.getElementById("locationCoords");
    const gpsStatus = document.getElementById("gpsStatus");
  
    // Simulate connecting to GPS module
    gpsStatus.textContent = "Connecting...";
    gpsStatus.style.backgroundColor = "rgba(255, 165, 0, 0.3)";
  
    setTimeout(() => {
      // Simulate GPS data received from ESP module
      const locations = [
        { lat: 51.5074, lng: -0.1278, name: "London" },
        { lat: 55.9533, lng: -3.1883, name: "Edinburgh" },
        { lat: 53.4808, lng: -2.2426, name: "Manchester" },
        { lat: 53.8008, lng: -1.5491, name: "Leeds" },
        { lat: 52.4862, lng: -1.8904, name: "Birmingham" },
      ];
  
      const randomLocation =
        locations[Math.floor(Math.random() * locations.length)];
  
      // Update the status
      gpsStatus.textContent = "Connected";
      gpsStatus.style.backgroundColor = "rgba(0, 255, 0, 0.3)";
  
      // Update location display
      locationCoords.innerHTML = `Your luggage is in <strong>${
        randomLocation.name
      }</strong><br>
                                   Coordinates: ${randomLocation.lat.toFixed(
                                     4
                                   )}, ${randomLocation.lng.toFixed(4)}`;
  
      // Add a magical effect to the map
      const mapContainer = document.getElementById("map");
      mapContainer.style.transition = "transform 0.5s, filter 0.5s";
      mapContainer.style.transform = "scale(0.95)";
      mapContainer.style.filter = "brightness(1.2) saturate(1.2)";
  
      setTimeout(() => {
        mapContainer.style.transform = "scale(1)";
        mapContainer.style.filter = "brightness(1) saturate(1)";
      }, 500);
    }, 2000);
  }
  
  // =================== FOLLOW ME FEATURE ===================
  function updateFollowStatus(isFollowing) {
    const followStatus = document.getElementById("followStatus");
  
    if (isFollowing) {
      // Simulate sending "ON" command to ESP
      followStatus.textContent = "Your luggage is now following you";
      followStatus.style.color = "#9369d9";
  
      // Add a subtle animation
      followStatus.style.animation = "pulse 2s infinite";
    } else {
      // Simulate sending "OFF" command to ESP
      followStatus.textContent = "Your luggage is staying put";
      followStatus.style.color = "white";
      followStatus.style.animation = "none";
    }
  }
  
  // =================== WEIGHT MONITOR FEATURE ===================
  function updateWeightDisplay() {
    const weightFill = document.getElementById("weightFill");
    const weightValue = document.getElementById("weightValue");
    const weightStatus = document.getElementById("weightStatus");
  
    // Simulate weight data from ESP/sensors
    const randomWeight = (Math.random() * 20 + 5).toFixed(1); // Between 5 and 25 kg
    const percentage = (randomWeight / 23) * 100;
  
    weightFill.style.width = `${percentage}%`;
    weightValue.textContent = `${randomWeight} kg`;
  
    // Update weight status message and color
    if (randomWeight > 23) {
      weightStatus.textContent = "Warning: Exceeds airline weight limit";
      weightFill.style.background = "linear-gradient(90deg, #ae0001, #eeba30)";
    } else if (randomWeight > 20) {
      weightStatus.textContent = "Approaching airline weight limit";
      weightFill.style.background = "linear-gradient(90deg, #eeba30, #ecb939)";
    } else if (randomWeight > 15) {
      weightStatus.textContent = "Weight within carry-on limits";
      weightFill.style.background = "linear-gradient(90deg, #1a472a, #2a623d)";
    } else {
      weightStatus.textContent = "Plenty of room for more items";
      weightFill.style.background = "linear-gradient(90deg, #3a1d6e, #9369d9)";
    }
  }
  
  // =================== HARRY POTTER QUOTES ===================
  function displayRandomQuote() {
    const quotes = [
      {
        text: "It does not do to dwell on dreams and forget to live.",
        author: "Albus Dumbledore",
      },
      {
        text: "Happiness can be found, even in the darkest of times, if one only remembers to turn on the light.",
        author: "Albus Dumbledore",
      },
      {
        text: "It takes a great deal of bravery to stand up to our enemies, but just as much to stand up to our friends.",
        author: "Albus Dumbledore",
      },
      {
        text: "Fear of a name only increases fear of the thing itself.",
        author: "Hermione Granger",
      },
      {
        text: "It is our choices that show what we truly are, far more than our abilities.",
        author: "Albus Dumbledore",
      },
      {
        text: "I solemnly swear that I am up to no good.",
        author: "The Marauder's Map",
      },
      { text: "After all this time? Always.", author: "Severus Snape" },
      {
        text: "We've all got both light and dark inside us. What matters is the part we choose to act on.",
        author: "Sirius Black",
      },
      {
        text: "Just because you have the emotional range of a teaspoon doesn't mean we all have.",
        author: "Hermione Granger",
      },
      { text: "Mischief managed!", author: "Harry Potter" },
    ];
  
    const randomIndex = Math.floor(Math.random() * quotes.length);
    const quote = quotes[randomIndex];
  
    document.getElementById("quoteText").textContent = quote.text;
    document.getElementById("quoteAuthor").textContent = `— ${quote.author}`;
  
    // Add a fade effect
    const quoteContainer = document.querySelector(".quote-container");
    quoteContainer.style.opacity = 0;
  
    setTimeout(() => {
      quoteContainer.style.transition = "opacity 1s";
      quoteContainer.style.opacity = 1;
    }, 300);
  }
  
  // =================== HOUSE SELECTION & THEMES ===================
  function setupHouseSelection() {
    // Make images clickable for house selection
    document.querySelectorAll(".house-option").forEach((option) => {
      option.style.cursor = "pointer";
    });
  }
  
  function applyHouseTheme(house) {
    // Remove all previous house classes
    document.body.classList.remove(
      "gryffindor-theme",
      "slytherin-theme",
      "ravenclaw-theme",
      "hufflepuff-theme"
    );
  
    // Apply the selected house theme
    document.body.classList.add(`${house}-theme`);
  
    // Update UI elements based on house
    updateUIForHouse(house);
  
    // Magical transition effect
    const main = document.querySelector("main");
    main.style.transition = "transform 0.5s, opacity 0.5s";
    main.style.transform = "scale(0.98)";
    main.style.opacity = "0.8";
  
    setTimeout(() => {
      main.style.transform = "scale(1)";
      main.style.opacity = "1";
    }, 500);
  }
  
  function updateUIForHouse(house) {
    const featureButtons = document.querySelectorAll(".feature-btn");
    const houseColors = {
      gryffindor: {
        primary: "#740001",
        secondary: "#ae0001",
        accent: "#eeba30",
      },
      slytherin: {
        primary: "#1a472a",
        secondary: "#2a623d",
        accent: "#aaaaaa",
      },
      ravenclaw: {
        primary: "#0e1a40",
        secondary: "#222f5b",
        accent: "#bebebe",
      },
      hufflepuff: {
        primary: "#ecb939",
        secondary: "#f0c75e",
        accent: "#726255",
      },
    };
  
    // Update button gradients for the selected house
    featureButtons.forEach((button) => {
      if (!button.classList.contains("btn-secondary")) {
        const colors = houseColors[house];
        button.style.background = `linear-gradient(135deg, ${colors.primary} 0%, ${colors.secondary} 100%)`;
      }
    });
  
    // Update active menu items
    const activeMenu = document.querySelector(".menuboxes.active");
    activeMenu.style.backgroundColor = `rgba(${getRGBValues(
      houseColors[house].primary
    )}, 0.5)`;
    activeMenu.style.borderColor = `rgba(${getRGBValues(
      houseColors[house].secondary
    )}, 0.8)`;
  }
  
  function getRGBValues(hex) {
    // Convert hex to RGB
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `${r}, ${g}, ${b}`;
  }
  
  function highlightSelectedHouse(house) {
    // Remove active class from all options
    document.querySelectorAll(".house-option").forEach((option) => {
      option.classList.remove("active");
    });
  
    // Add active class to selected house
    document
      .querySelector(`.house-option[data-house="${house}"]`)
      .classList.add("active");
  }
  
  // =================== PASSWORD MANAGEMENT ===================
  function updatePassword() {
    const currentPassword = document.getElementById("currentPassword").value;
    const newPassword = document.getElementById("newPassword").value;
    const confirmPassword = document.getElementById("confirmPassword").value;
  
    // Simple validation
    if (!currentPassword || !newPassword || !confirmPassword) {
      showPasswordNotification("Please fill in all password fields", false);
      return;
    }
  
    if (newPassword !== confirmPassword) {
      showPasswordNotification("New passwords do not match", false);
      return;
    }
  
    // Here you would normally validate against the current password
    // and then save the new password securely
  
    // For demo purposes, we'll just simulate success
    showPasswordNotification("Password updated successfully!", true);
  
    // Clear the fields
    document.getElementById("currentPassword").value = "";
    document.getElementById("newPassword").value = "";
    document.getElementById("confirmPassword").value = "";
  }
  
  function resetPassword() {
    // Simulate sending a reset email
    showPasswordNotification(
      "Password reset instructions sent to your email",
      true
    );
  }
  
  function showPasswordNotification(message, isSuccess) {
    // Create notification element if it doesn't exist
    let notification = document.getElementById("passwordNotification");
    if (!notification) {
      notification = document.createElement("div");
      notification.id = "passwordNotification";
      notification.style.padding = "10px 15px";
      notification.style.borderRadius = "8px";
      notification.style.marginTop = "15px";
      notification.style.textAlign = "center";
      notification.style.transition = "opacity 0.5s";
  
      const securitySection = document.querySelector(
        ".feature-card:nth-child(2) .feature-content"
      );
      securitySection.appendChild(notification);
    }
  
    // Set appearance based on success/failure
    if (isSuccess) {
      notification.style.backgroundColor = "rgba(0, 255, 0, 0.2)";
      notification.style.border = "1px solid rgba(0, 255, 0, 0.5)";
    } else {
      notification.style.backgroundColor = "rgba(255, 0, 0, 0.2)";
      notification.style.border = "1px solid rgba(255, 0, 0, 0.5)";
    }
  
    // Show message
    notification.textContent = message;
    notification.style.opacity = "1";
  
    // Hide after 5 seconds
    setTimeout(() => {
      notification.style.opacity = "0";
    }, 5000);
  }
  
  // =================== SORTING HAT ===================
  function consultSortingHat() {
    const sortingResult = document.getElementById("sortingResult");
    sortingResult.textContent = "The Sorting Hat is thinking...";
    sortingResult.className = "sorting-result active";
  
    // Simulate the Sorting Hat's deliberation
    setTimeout(() => {
      const houses = ["gryffindor", "slytherin", "ravenclaw", "hufflepuff"];
      const randomHouse = houses[Math.floor(Math.random() * houses.length)];
      const houseNames = {
        gryffindor: "Gryffindor",
        slytherin: "Slytherin",
        ravenclaw: "Ravenclaw",
        hufflepuff: "Hufflepuff",
      };
  
      // Update sorting result
      sortingResult.textContent = `The Sorting Hat has decided: ${houseNames[randomHouse]}!`;
      sortingResult.className = `sorting-result active ${randomHouse}`;
  
      // Apply the house theme automatically
      applyHouseTheme(randomHouse);
      highlightSelectedHouse(randomHouse);
      localStorage.setItem("selectedHouse", randomHouse);
  
      // Add special effects
      const sortingHatBtn = document.getElementById("sortingHat");
      sortingHatBtn.classList.add("inactive");
      sortingHatBtn.textContent = "Sorted!";
  
      // Re-enable after 3 seconds
      setTimeout(() => {
        sortingHatBtn.classList.remove("inactive");
        sortingHatBtn.textContent = "Consult the Sorting Hat";
      }, 3000);
    }, 2000);
  }
  
  // =================== ANIMATIONS ===================
  // Add a pulse animation for the Follow Me feature
  const styleSheet = document.createElement("style");
  styleSheet.textContent = `
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    
    .feature-btn.inactive {
        background: rgba(100, 100, 100, 0.5) !important;
        cursor: not-allowed;
    }
  `;
  document.head.appendChild(styleSheet);
  
  // =================== SIMULATED BACKEND UPDATES ===================
  // Periodically update information to simulate a real system
  setInterval(() => {
    // Randomly decide which feature to update
    const randomFeature = Math.floor(Math.random() * 3);
  
    switch (randomFeature) {
      case 0:
        // Sometimes update location
        if (Math.random() < 0.3) {
          updateGPSLocation();
        }
        break;
      case 1:
        // Sometimes update weight
        if (Math.random() < 0.5) {
          updateWeightDisplay();
        }
        break;
      case 2:
        // Rarely show a new quote automatically
        if (Math.random() < 0.2) {
          displayRandomQuote();
        }
        break;
    }
  }, 30000); // Every 30 seconds