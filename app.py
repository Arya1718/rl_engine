from __future__ import annotations

import json
import traceback

from flask import Flask, jsonify, render_template, request

from rl.simulation import (
    ALGORITHMS,
    explain_counterfactual,
    run_ablation_studio,
    run_comparison,
    run_comparison_stream,
    run_convergence_diagnostics,
    run_scenario_lab,
    run_sensitivity_analysis,
    run_policy_inspection,
    extract_pareto_frontier,
    record_run,
    get_run_history,
)

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.get("/")
def index():
    return render_template("index.html", algorithms=ALGORITHMS)


@app.get("/api/algorithms")
def algorithms():
    return jsonify({"algorithms": ALGORITHMS})


@app.post("/api/simulate")
def simulate():
    try:
        payload = request.get_json(silent=True) or {}
        result = run_comparison(payload)
        record_run(result)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.post("/api/ablation")
def ablation():
    try:
        payload = request.get_json(silent=True) or {}
        result = run_ablation_studio(payload)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.post("/api/scenario_lab")
def scenario_lab():
    try:
        payload = request.get_json(silent=True) or {}
        result = run_scenario_lab(payload)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.post("/api/counterfactual")
def counterfactual():
    try:
        payload = request.get_json(silent=True) or {}
        result = explain_counterfactual(payload)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.post("/api/sensitivity")
def sensitivity():
    try:
        payload = request.get_json(silent=True) or {}
        result = run_sensitivity_analysis(payload)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.post("/api/convergence")
def convergence():
    try:
        payload = request.get_json(silent=True) or {}
        result = run_convergence_diagnostics(payload)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.get("/api/simulate_stream")
def simulate_stream():
    payload_raw = request.args.get("payload", "{}")
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        payload = {}

    def event_stream():
        try:
            for event in run_comparison_stream(payload):
                event_type = event.get("type", "message")
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(event)}\n\n"
        except GeneratorExit:
            return
        except Exception as exc:
            event = {
                "type": "stream_error",
                "message": str(exc),
            }
            yield "event: stream_error\n"
            yield f"data: {json.dumps(event)}\n\n"

    return app.response_class(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/policy_inspect")
def policy_inspect():
    try:
        payload = request.get_json(silent=True) or {}
        result = run_policy_inspection(payload)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.post("/api/pareto")
def pareto():
    try:
        payload = request.get_json(silent=True) or {}
        result = extract_pareto_frontier(payload)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.get("/api/run_history")
def run_history():
    return jsonify({"history": get_run_history()})


if __name__ == "__main__":
    app.run(debug=True)
