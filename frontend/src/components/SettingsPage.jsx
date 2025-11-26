import { useEffect, useMemo, useState } from 'react';
import { api } from '../api';
import './SettingsPage.css';

export default function SettingsPage({ onClose }) {
  const [settings, setSettings] = useState(null);
  const [ollamaModels, setOllamaModels] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const isOllama = settings?.llm_provider === 'ollama';

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    setIsLoading(true);
    setError('');
    setSuccess('');
    try {
      const data = await api.getSettings();
      setSettings({ ...data, council_models: data.council_models || [] });
      if (data.llm_provider === 'ollama') {
        await loadOllamaModels();
      }
    } catch (err) {
      setError(err.message || 'Failed to load settings');
    } finally {
      setIsLoading(false);
    }
  };

  const loadOllamaModels = async () => {
    try {
      const models = await api.listOllamaModels();
      setOllamaModels(models);
    } catch (err) {
      setError(err.message || 'Unable to load Ollama models');
    }
  };

  const toggleCouncilModel = (modelName) => {
    setSettings((prev) => {
      const selected = new Set(prev.council_models || []);
      if (selected.has(modelName)) {
        selected.delete(modelName);
      } else {
        selected.add(modelName);
      }
      return { ...prev, council_models: Array.from(selected) };
    });
  };

  const handleProviderChange = (provider) => {
    setSettings((prev) => ({ ...prev, llm_provider: provider }));
    if (provider === 'ollama' && ollamaModels.length === 0) {
      loadOllamaModels();
    }
  };

  const handleTextCouncilChange = (value) => {
    const models = value
      .split(',')
      .map((m) => m.trim())
      .filter(Boolean);
    setSettings((prev) => ({ ...prev, council_models: models }));
  };

  const handleSave = async () => {
    if (!settings) return;
    setIsSaving(true);
    setError('');
    setSuccess('');
    try {
      const updated = await api.updateSettings(settings);
      setSettings({ ...updated, council_models: updated.council_models || [] });
      setSuccess('Settings saved successfully');
    } catch (err) {
      setError(err.message || 'Failed to save settings');
    } finally {
      setIsSaving(false);
    }
  };

  const ollamaModelNames = useMemo(
    () => (ollamaModels || []).map((model) => model.name).filter(Boolean),
    [ollamaModels]
  );

  if (isLoading || !settings) {
    return (
      <div className="settings-page">
        <div className="settings-card">
          <div className="settings-header">
            <h2>Settings</h2>
          </div>
          <div className="settings-loading">Loading settings...</div>
        </div>
      </div>
    );
  }

  const councilTextValue = (settings.council_models || []).join(', ');

  return (
    <div className="settings-page">
      <div className="settings-card">
        <div className="settings-header">
          <div>
            <div className="eyebrow">Administration</div>
            <h2>LLM Provider & Models</h2>
            <p className="muted">
              Choose which provider powers the council and, when using Ollama, pick the local models to include
              in deliberations and the model that acts as the judge.
            </p>
          </div>
          <div className="actions">
            <button className="secondary" onClick={onClose}>
              Back to chat
            </button>
            <button className="primary" onClick={handleSave} disabled={isSaving}>
              {isSaving ? 'Saving…' : 'Save changes'}
            </button>
          </div>
        </div>

        {error && <div className="alert error">{error}</div>}
        {success && <div className="alert success">{success}</div>}

        <div className="form-group">
          <label htmlFor="provider">Provider</label>
          <select
            id="provider"
            value={settings.llm_provider}
            onChange={(e) => handleProviderChange(e.target.value)}
          >
            <option value="openrouter">OpenRouter</option>
            <option value="ollama">Ollama</option>
          </select>
        </div>

        {isOllama ? (
          <>
            <div className="form-group inline">
              <div>
                <label>Available Ollama models</label>
                <p className="muted">Select which models participate in the council.</p>
              </div>
              <button className="secondary" onClick={loadOllamaModels}>
                Refresh list
              </button>
            </div>

            <div className="model-grid">
              {ollamaModelNames.length === 0 ? (
                <div className="muted">No models found on the Ollama host.</div>
              ) : (
                ollamaModelNames.map((name) => (
                  <label key={name} className="model-option">
                    <input
                      type="checkbox"
                      checked={settings.council_models?.includes(name)}
                      onChange={() => toggleCouncilModel(name)}
                    />
                    <span>{name}</span>
                  </label>
                ))
              )}
            </div>

            <div className="form-group">
              <label htmlFor="judge">Judge model</label>
              <select
                id="judge"
                value={settings.chairman_model || ''}
                onChange={(e) => setSettings((prev) => ({ ...prev, chairman_model: e.target.value }))}
              >
                <option value="">Select a model</option>
                {ollamaModelNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
                {settings.chairman_model && !ollamaModelNames.includes(settings.chairman_model) && (
                  <option value={settings.chairman_model}>{settings.chairman_model} (current)</option>
                )}
              </select>
              <p className="muted">Used for final synthesis and judging rankings.</p>
            </div>
          </>
        ) : (
          <>
            <div className="form-group">
              <label htmlFor="council-models">Council models (comma-separated)</label>
              <input
                id="council-models"
                type="text"
                value={councilTextValue}
                onChange={(e) => handleTextCouncilChange(e.target.value)}
                placeholder="openai/gpt-5.1, google/gemini-3-pro-preview"
              />
            </div>

            <div className="form-group">
              <label htmlFor="judge">Judge model</label>
              <input
                id="judge"
                type="text"
                value={settings.chairman_model || ''}
                onChange={(e) => setSettings((prev) => ({ ...prev, chairman_model: e.target.value }))}
                placeholder="google/gemini-3-pro-preview"
              />
            </div>
          </>
        )}

        <div className="form-group">
          <label htmlFor="title-model">Title generation model</label>
          {isOllama ? (
            <select
              id="title-model"
              value={settings.title_model || ''}
              onChange={(e) => setSettings((prev) => ({ ...prev, title_model: e.target.value }))}
            >
              <option value="">Select a model</option>
              {ollamaModelNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
              {settings.title_model && !ollamaModelNames.includes(settings.title_model) && (
                <option value={settings.title_model}>{settings.title_model} (current)</option>
              )}
            </select>
          ) : (
            <input
              id="title-model"
              type="text"
              value={settings.title_model || ''}
              onChange={(e) => setSettings((prev) => ({ ...prev, title_model: e.target.value }))}
              placeholder="google/gemini-2.5-flash"
            />
          )}
          <p className="muted">Controls the short titles shown in the sidebar.</p>
        </div>
      </div>
    </div>
  );
}
