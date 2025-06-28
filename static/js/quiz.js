document.addEventListener("DOMContentLoaded", function () {

  console.log("Hey there user");
    // Quiz data
    const quizData = [
      {
        question:
          "When packing for a journey, which approach describes you best?",
        options: [
          "I plan everything meticulously weeks in advance",
          "I pack light and only bring essentials",
          "I bring everything I might possibly need, just in case",
          "I usually pack at the last minute but somehow make it work",
        ],
      },
      {
        question: "What's most important to you in luggage?",
        options: [
          "Durability and protection for my belongings",
          "Lightweight and easy to carry around",
          "Maximum storage space and organization",
          "Style and unique appearance",
        ],
      },
      {
        question: "When faced with a travel challenge, you typically:",
        options: [
          "Research thoroughly and prepare for all possibilities",
          "Adapt quickly and find creative solutions",
          "Rely on having packed the right tools for any situation",
          "Ask locals or fellow travelers for advice",
        ],
      },
      {
        question: "Your ideal travel destination would be:",
        options: [
          "A historical city with museums and cultural landmarks",
          "An outdoor adventure in nature",
          "A comfortable resort with amenities",
          "An unexplored location off the beaten path",
        ],
      },
      {
        question: "Which magical feature would you most value in your luggage?",
        options: [
          "Anti-theft protection and security spells",
          "Self-lightening charm that makes it weightless",
          "Undetectable extension charm for unlimited space",
          "Self-navigation to always find its way back to you",
        ],
      },
    ];
  
    // Luggage personalities with their descriptions
    const luggageTypes = [
      {
        name: "The Sentinel Trunk",
        description:
          "Protected by powerful guardian charms, this trunk is perfect for the meticulous planner who values security and organization above all. Its reinforced structure and magical locks keep your belongings safe in any circumstance.",
        image: "sentinel-trunk.jpg",
      },
      {
        name: "The Nomad Backpack",
        description:
          "Lightweight and adaptable, this enchanted backpack is ideal for the minimalist traveler. With terrain-adjusting straps and weather-resistant fabrics, it's the perfect companion for any adventure.",
        image: "nomad-backpack.jpg",
      },
      {
        name: "The Cornucopia Case",
        description:
          "Featuring an undetectable extension charm, this deceptively spacious case is for those who like to be prepared for anything. Multiple compartments organize your belongings while maintaining a reasonable exterior size.",
        image: "cornucopia-case.jpg",
      },
      {
        name: "The Pathfinder Satchel",
        description:
          "This intuitive satchel seems to anticipate your needs before you do. With a knack for rearranging its contents to bring forward exactly what you're looking for, it's perfect for the spontaneous traveler.",
        image: "pathfinder-satchel.jpg",
      },
    ];
  
    // Store DOM elements
    const quizContainer = document.querySelector(".quiz-container");
  
    // Show initial options
    function showInitialOptions() {
      const initialOptionsHTML = `
          <div class="welcome-container">
            <h2 class="welcome-title">Welcome to Magical Luggage Companion</h2>
            <p class="welcome-description">What would you like to do today?</p>
            
            <div class="initial-options-container">
              <button class="option-btn find-new-btn">
                <span class="option-icon">🧳</span>
                <span class="option-label">Find My Perfect Luggage</span>
                <p class="option-description">Take our magical quiz to discover which enchanted luggage matches your travel personality</p>
              </button>
              
              <button class="option-btn view-features-btn">
                <span class="option-icon">✨</span>
                <span class="option-label">View My Suitcase Features</span>
                <p class="option-description">Explore the magical features of your existing enchanted luggage</p>
              </button>
            </div>
          </div>
        `;
  
      quizContainer.innerHTML = initialOptionsHTML;
  
      // Add event listeners
      document
        .querySelector(".find-new-btn")
        .addEventListener("click", startQuiz);
      document
        .querySelector(".view-features-btn")
        .addEventListener("click", viewFeatures);
  
      // Apply custom styles
      addInitialStyles();
    }
  
    // View features of existing suitcase
    function viewFeatures() {
      window.location.href = "/features";
    }
  
    // Start the quiz
    function startQuiz() {
      setupQuizInterface();
      loadQuestion();
      updateProgress();
  
      // Add event listeners for navigation buttons
      document
        .querySelector(".prev-btn")
        .addEventListener("click", goToPrevQuestion);
      document
        .querySelector(".next-btn")
        .addEventListener("click", goToNextQuestion);
    }
  
    // Set up the quiz interface
    function setupQuizInterface() {
      const quizInterfaceHTML = `
          <!-- Sorting Hat Section - Added this section -->
          <div class="sorting-hat">
            <div class="hat-image">
              <img src="sorting.png" alt="Sorting Hat" />
            </div>
            <h2 class="sorting-title">The Luggage Sorting Quiz</h2>
            <p class="sorting-subtitle">
              Answer the questions honestly, and we'll find your perfect magical
              travel companion!
            </p>
          </div>
          
          <div class="quiz-header">
            <div class="progress-container">
              <div class="progress"></div>
            </div>
            <div class="progress-text">Question 1/5</div>
          </div>
          <div class="question-container">
            <h2 class="question-text"></h2>
            <div class="options-container"></div>
          </div>
          <div class="navigation">
            <button class="quiz-btn prev-btn" disabled>Previous</button>
            <button class="quiz-btn next-btn" disabled>Next Question</button>
          </div>
        `;
  
      quizContainer.innerHTML = quizInterfaceHTML;
    }
  
    // Store state variables
    let currentQuestion = 0;
    let selectedAnswers = [];
  
    // Load question and options
    function loadQuestion() {
      const currentQuizData = quizData[currentQuestion];
      const questionText = document.querySelector(".question-text");
      const optionsContainer = document.querySelector(".options-container");
  
      questionText.textContent = currentQuizData.question;
  
      optionsContainer.innerHTML = "";
  
      currentQuizData.options.forEach((option, index) => {
        const isChecked =
          selectedAnswers[currentQuestion] === index ? "checked" : "";
  
        const optionHTML = `
            <div class="option">
              <input type="radio" name="question${currentQuestion}" id="option${index}" ${isChecked} />
              <label for="option${index}">
                <span class="checkmark"></span>
                <span class="option-text">${option}</span>
              </label>
            </div>
          `;
  
        optionsContainer.insertAdjacentHTML("beforeend", optionHTML);
      });
  
      // Add event listeners to options
      document
        .querySelectorAll(`input[name="question${currentQuestion}"]`)
        .forEach((input, index) => {
          input.addEventListener("change", () => {
            selectedAnswers[currentQuestion] = index;
            updateButtonStates();
          });
        });
  
      updateButtonStates();
    }
  
    // Update progress bar and text
    function updateProgress() {
      const progressBar = document.querySelector(".progress");
      const progressText = document.querySelector(".progress-text");
      const progressPercentage = ((currentQuestion + 1) / quizData.length) * 100;
  
      progressBar.style.width = `${progressPercentage}%`;
      progressText.textContent = `Question ${currentQuestion + 1}/${
        quizData.length
      }`;
    }
  
    // Update button states based on current question and selections
    function updateButtonStates() {
      const prevBtn = document.querySelector(".prev-btn");
      const nextBtn = document.querySelector(".next-btn");
  
      prevBtn.disabled = currentQuestion === 0;
  
      if (currentQuestion === quizData.length - 1) {
        nextBtn.textContent = "See Results";
        // Only enable if an option is selected
        nextBtn.disabled = selectedAnswers[currentQuestion] === undefined;
      } else {
        nextBtn.textContent = "Next Question";
        // Only enable if an option is selected
        nextBtn.disabled = selectedAnswers[currentQuestion] === undefined;
      }
    }
  
    // Go to previous question
    function goToPrevQuestion() {
      if (currentQuestion > 0) {
        currentQuestion--;
        loadQuestion();
        updateProgress();
      }
    }
  
    // Go to next question or show results
    function goToNextQuestion() {
      if (currentQuestion < quizData.length - 1) {
        currentQuestion++;
        loadQuestion();
        updateProgress();
      } else {
        showResults();
      }
    }
  
    // Calculate and show results
    function showResults() {
      // Count the frequency of each answer type
      const answerCounts = [0, 0, 0, 0];
  
      selectedAnswers.forEach((answer) => {
        answerCounts[answer]++;
      });
  
      // Find the most common answer (personality type)
      let maxCount = 0;
      let personalityIndex = 0;
  
      answerCounts.forEach((count, index) => {
        if (count > maxCount) {
          maxCount = count;
          personalityIndex = index;
        }
      });
  
      const resultLuggage = luggageTypes[personalityIndex];
  
      // Create results HTML
      const resultsHTML = `
          <div class="results-container">
            <h2 class="sorting-title">Your Perfect Match</h2>
            <div class="luggage-image">
              <img src="/api/placeholder/200/200" alt="${resultLuggage.name}" />
            </div>
            <h3 class="luggage-name">${resultLuggage.name}</h3>
            <p class="luggage-description">${resultLuggage.description}</p>
            <button class="quiz-btn restart-btn">Take Quiz Again</button>
            <button class="quiz-btn shop-btn">Shop ${resultLuggage.name}</button>
          </div>
        `;
  
      // Replace quiz content with results
      quizContainer.innerHTML = resultsHTML;
  
      // Add event listener for restart button
      document.querySelector(".restart-btn").addEventListener("click", () => {
        window.location.reload();
      });
  
      // Add event listener for shop button
      document.querySelector(".shop-btn").addEventListener("click", () => {
        window.location.href = "shop.html";
      });
    }
  
    // Add custom styles for initial options
    function addInitialStyles() {
      const style = document.createElement("style");
      style.textContent = `
          .welcome-container {
            text-align: center;
            padding: 2rem;
            animation: fadeIn 1s ease;
          }
          
          .welcome-title {
            font-family: "Cinzel", serif;
            font-size: 2.4rem;
            margin-bottom: 1rem;
            color: #9369d9;
            text-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
          }
          
          .welcome-description {
            font-family: "Mulish", sans-serif;
            font-size: 1.2rem;
            margin-bottom: 2.5rem;
            color: rgba(255, 255, 255, 0.9);
          }
          
          .initial-options-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            max-width: 600px;
            margin: 0 auto;
          }
          
          .option-btn {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
          }
          
          .option-btn:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
          }
          
          .option-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
          }
          
          .option-label {
            font-family: "Cinzel", serif;
            font-size: 1.5rem;
            margin-bottom: 0.8rem;
            color: #9369d9;
          }
          
          .option-description {
            font-family: "Mulish", sans-serif;
            font-size: 1rem;
            line-height: 1.6;
            color: rgba(255, 255, 255, 0.8);
            margin: 0;
          }
          
          .results-container {
            text-align: center;
            animation: fadeIn 1s ease;
          }
          
          .luggage-image {
            width: 200px;
            height: 200px;
            margin: 2rem auto;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
          }
          
          .luggage-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
          }
          
          .luggage-name {
            font-family: "Cinzel", serif;
            font-size: 2.2rem;
            margin-bottom: 1.5rem;
            color: #9369d9;
            text-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
          }
          
          .luggage-description {
            font-family: "Mulish", sans-serif;
            font-size: 1.2rem;
            line-height: 1.8;
            margin-bottom: 2.5rem;
            color: rgba(255, 255, 255, 0.9);
          }
          
          .shop-btn {
            background: linear-gradient(135deg, #6a3cb3 0%, #9369d9 100%);
            margin-left: 1rem;
          }
          
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
          }
        `;
  
      document.head.appendChild(style);
    }
  
    // Initialize with welcome screen
    showInitialOptions();
  });