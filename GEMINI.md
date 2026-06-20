Here is the exact, code-free prompt to feed your coding agent to surgically address the historical proximity issue.

---

### **Agent Implementation Prompt: Historical Proximity Fix**

**Objective:** Refactor the historical lineage selector (`_select_deepest_historical_set_id`) in `src/services/fda_client.py` to anchor its selection to the patient's prescription date. The current logic risks selecting defunct lineages that miss years of updates because it searches for the absolute oldest `v1` date rather than the date closest to the prescription. Do not generate explanations; simply implement the following logical flow.

---

#### **Required Modifications in `src/services/fda_client.py`:**

**1. Update Function Signatures & Data Flow**

* Modify the signature of `_select_deepest_historical_set_id` to accept the parsed prescription date (as a `datetime` object) as an additional parameter.
* Update the orchestrator function (`get_past_and_present_labels`) to correctly pass the parsed prescription date down to this historical selector.

**2. Preserve the Pre-Sort Network Guard**

* Do not change the initial candidate array slice. Keep the logic that pre-sorts by total version count and slices the top `_HISTORICAL_PROBE_LIMIT` candidates to prevent excessive network calls.

**3. Refactor the Concurrent Probe Logic**

* Rename the inner asynchronous helper from `probe_v1_date` to something reflecting its new purpose (e.g., `probe_closest_date`).
* Inside this helper, fetch the candidate's version history ledger.
* Instead of looking for `spl_version == 1`, iterate through the history entries and parse their publication dates.
* **The Core Logic:** Filter out any dates that are strictly *greater* than the prescription date. Find the maximum date from the remaining entries (i.e., the version that was active exactly on the day the prescription was written).
* Return a tuple containing the candidate's `setid` and this calculated closest date.

**4. Implement Sentinel Fallbacks**

* If a candidate has no versions prior to the prescription date, or if the HTTP request fails, gracefully catch the exception.
* Return a sentinel date located far in the past (e.g., the year 1000) so that this invalid candidate naturally sinks to the bottom of the final sorting step.

**5. Update the Final Sorting Step**

* Once the concurrent probes return their tuples, sort the results based on the closest valid dates.
* The sort must be in **descending order** so that the candidate whose active label is mathematically closest to the prescription date wins index 0.
* Return the `setid` of this winning candidate.