/**
 * Mini-Chat-Terminal für das eigene Modell (Ink = React fürs Terminal).
 *
 * Voraussetzung: der Inferenz-Server läuft →  python chat_server.py
 * Start:                                      npm start
 *
 * Das Modell ist ein BASISMODELL: es setzt Text fort, es antwortet nicht.
 * Prompts also als Satzanfänge formulieren ("Die Hauptstadt von Frankreich"),
 * nicht als Fragen — bis die SFT-Version da ist. ;)
 */
import React, { useEffect, useState } from 'react';
import { render, Box, Text, useApp, useInput } from 'ink';
import TextInput from 'ink-text-input';
import Spinner from 'ink-spinner';

const SERVER = process.env.SERVER ?? 'http://127.0.0.1:8123';

function App() {
	const { exit } = useApp();
	const [info, setInfo] = useState(null);
	const [infoError, setInfoError] = useState(null);
	const [input, setInput] = useState('');
	const [history, setHistory] = useState([]); // {prompt, text, stats | error}
	const [busy, setBusy] = useState(false);

	// Modell-Info beim Start holen — dient gleichzeitig als Server-Check.
	useEffect(() => {
		fetch(`${SERVER}/info`)
			.then((r) => r.json())
			.then(setInfo)
			.catch(() => setInfoError(
				`Kein Server unter ${SERVER} — erst starten: python chat_server.py`,
			));
	}, []);

	useInput((_ch, key) => {
		if (key.escape) exit();
	});

	const submit = async (prompt) => {
		if (!prompt.trim() || busy) return;
		setInput('');
		setBusy(true);
		try {
			const res = await fetch(`${SERVER}/generate`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ prompt }),
			});
			const data = await res.json();
			setHistory((h) => [...h, {
				prompt,
				text: data.text,
				stats: `${data.seconds}s · ${data.tokens_per_s} tok/s`,
			}]);
		} catch (e) {
			setHistory((h) => [...h, { prompt, error: String(e) }]);
		} finally {
			setBusy(false);
		}
	};

	const mio = info ? Math.round(info.params / 1e6) : null;

	return (
		<Box flexDirection="column" paddingX={1}>
			<Box borderStyle="round" borderColor="cyan" paddingX={1}>
				<Text color="cyan" bold>
					{info
						? `Dein Modell · ${mio}M Parameter · ${info.device} · ${info.checkpoint}`
						: infoError ?? 'Verbinde mit Server...'}
				</Text>
			</Box>

			{history.map((entry, i) => (
				<Box key={i} flexDirection="column" marginTop={1}>
					<Text color="yellow" bold>{'Du     › '}<Text color="white">{entry.prompt}</Text></Text>
					{entry.error
						? <Text color="red">Fehler: {entry.error}</Text>
						: (
							<>
								<Text color="green" bold>{'Modell › '}<Text color="white">{entry.text}</Text></Text>
								<Text dimColor>         {entry.stats}</Text>
							</>
						)}
				</Box>
			))}

			<Box marginTop={1}>
				{busy ? (
					<Text color="green"><Spinner type="dots" /> generiert...</Text>
				) : (
					<>
						<Text color="yellow" bold>{'Du     › '}</Text>
						<TextInput
							value={input}
							onChange={setInput}
							onSubmit={submit}
							placeholder="Satzanfang eingeben (Enter sendet, Esc beendet)"
						/>
					</>
				)}
			</Box>
		</Box>
	);
}

if (!process.stdin.isTTY) {
	console.error('Braucht ein echtes Terminal (TTY) — bitte direkt in der Shell starten.');
	process.exit(1);
}
render(<App />);
