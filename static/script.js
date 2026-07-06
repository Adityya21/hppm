/**
 * script.js — House Price Prediction Frontend
 * =============================================
 * Handles form submission (AJAX), slider sync,
 * input validation, and result display.
 */

document.addEventListener("DOMContentLoaded", () => {

    // ====================================================================
    // Element References
    // ====================================================================
    const form          = document.getElementById("prediction-form");
    const btnPredict    = document.getElementById("btn-predict");
    const resultContent = document.getElementById("result-content");
    const resultError   = document.getElementById("result-error");
    const placeholder   = document.getElementById("result-placeholder");

    const resultPrice    = document.getElementById("result-price");
    const detailLocation = document.getElementById("detail-location");
    const detailSqft     = document.getElementById("detail-sqft");
    const detailBath     = document.getElementById("detail-bath");
    const detailBedrooms = document.getElementById("detail-bedrooms");
    const errorMessage   = document.getElementById("error-message");

    const sqftSlider    = document.getElementById("sqft-slider");
    const sqftInput     = document.getElementById("total_sqft");
    const sqftValue     = document.getElementById("sqft-value");

    const bathSlider    = document.getElementById("bath-slider");
    const bathInput     = document.getElementById("bath");
    const bathValue     = document.getElementById("bath-value");

    const bedroomSlider = document.getElementById("bedroom-slider");
    const bedroomInput  = document.getElementById("bedrooms");
    const bedroomValue  = document.getElementById("bedroom-value");

    // ====================================================================
    // Slider ↔ Input Sync
    // ====================================================================
    function syncSlider(slider, input, display) {
        if (!slider || !input) return;

        slider.addEventListener("input", () => {
            input.value = slider.value;
            if (display) display.textContent = formatNumber(slider.value);
        });

        input.addEventListener("input", () => {
            let val = parseFloat(input.value) || parseFloat(slider.min);
            val = Math.max(parseFloat(slider.min), Math.min(parseFloat(slider.max), val));
            slider.value = val;
            if (display) display.textContent = formatNumber(val);
        });
    }

    function formatNumber(val) {
        const n = parseFloat(val);
        return n >= 1000 ? n.toLocaleString("en-IN") : n.toString();
    }

    syncSlider(sqftSlider, sqftInput, sqftValue);
    syncSlider(bathSlider, bathInput, bathValue);
    syncSlider(bedroomSlider, bedroomInput, bedroomValue);

    // ====================================================================
    // Form Submission
    // ====================================================================
    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            if (!validate()) return;

            btnPredict.classList.add("loading");
            hideResults();

            try {
                const formData = new FormData(form);
                const data = Object.fromEntries(formData.entries());

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data),
                });

                const result = await response.json();

                if (response.ok && result.success) {
                    showSuccess(result);
                } else {
                    showError(result.error || "Prediction failed. Please try again.");
                }
            } catch (err) {
                console.error("Prediction error:", err);
                showError("Network error. Please check your connection.");
            } finally {
                btnPredict.classList.remove("loading");
            }
        });
    }

    // ====================================================================
    // Result Display
    // ====================================================================
    function showSuccess(result) {
        const price = result.predicted_price;
        resultPrice.textContent = `₹ ${price.toLocaleString("en-IN", {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })} Lakhs`;

        detailLocation.textContent = result.input.location;
        detailSqft.textContent = parseFloat(result.input.total_sqft).toLocaleString("en-IN") + " sq.ft";
        detailBath.textContent = result.input.bath;
        detailBedrooms.textContent = result.input.bedrooms;

        placeholder.style.display = "none";
        resultError.classList.remove("visible");
        resultContent.classList.add("visible");

        // Scroll to result on mobile
        if (window.innerWidth <= 900) {
            resultContent.scrollIntoView({ behavior: "smooth", block: "center" });
        }
    }

    function showError(message) {
        errorMessage.textContent = message;
        placeholder.style.display = "none";
        resultContent.classList.remove("visible");
        resultError.classList.add("visible");
    }

    function hideResults() {
        resultContent.classList.remove("visible");
        resultError.classList.remove("visible");
    }

    // ====================================================================
    // Validation
    // ====================================================================
    function validate() {
        const location = document.getElementById("location").value;
        const sqft     = parseFloat(sqftInput.value);
        const bath     = parseFloat(bathInput.value);
        const bedrooms = parseInt(bedroomInput.value);

        if (!location) { showError("Please select a location."); return false; }
        if (isNaN(sqft) || sqft <= 0) { showError("Please enter a valid area."); return false; }
        if (isNaN(bath) || bath <= 0) { showError("Please enter bathrooms."); return false; }
        if (isNaN(bedrooms) || bedrooms <= 0) { showError("Please enter bedrooms."); return false; }
        return true;
    }

    // ====================================================================
    // Location Search Filter
    // ====================================================================
    const locationSelect = document.getElementById("location");
    const locationSearch = document.getElementById("location-search");

    if (locationSearch && locationSelect) {
        locationSearch.addEventListener("input", () => {
            const query = locationSearch.value.toLowerCase().trim();
            const options = locationSelect.querySelectorAll("option");
            options.forEach((opt) => {
                if (opt.value === "") { opt.style.display = ""; return; }
                opt.style.display = opt.textContent.toLowerCase().includes(query) ? "" : "none";
            });
        });
    }
});
