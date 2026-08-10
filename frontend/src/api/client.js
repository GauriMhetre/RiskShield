/**
 * API client for RiskShield backend.
 *
 * This is the ONE place all API calls are made from. Every component imports
 * from here rather than calling fetch() directly, ensuring:
 * - Consistent base URL and error handling across the app
 * - Easy to update headers, auth, or logging in one place
 * - In a later phase, this becomes environment-configurable (not hardcoded)
 */

// API base URL — will become an environment variable in a later phase (Phase 8/9)
// For now, hardcoded to match the FastAPI backend's default development port
const API_BASE_URL = "http://localhost:8000";

/**
 * Fetch flagged transactions from the backend.
 *
 * @param {Object} options - Query parameters (all optional)
 * @param {string} options.since - ISO 8601 datetime to filter transactions since this time
 * @param {number} options.minScore - Minimum risk score (0-1) to filter by
 * @param {number} options.limit - Maximum number of results to return
 * @returns {Promise<Array>} - Array of flagged transaction objects
 * @throws {Error} - If the API request fails, throws an error with the HTTP status code
 */
export async function getFlags({ since, minScore, limit } = {}) {
  try {
    // Build query string from provided parameters
    const params = new URLSearchParams();
    if (since) params.append("since", since);
    if (minScore !== undefined) params.append("min_score", minScore);
    if (limit !== undefined) params.append("limit", limit);

    const queryString = params.toString();
    const url = `${API_BASE_URL}/flags${queryString ? `?${queryString}` : ""}`;

    const response = await fetch(url);

    // Check if the response is OK (status 200-299)
    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    // Parse and return the JSON response
    const data = await response.json();
    return data;
  } catch (error) {
    // Re-throw with clear messaging for debugging
    console.error("getFlags() error:", error);
    throw error;
  }
}
