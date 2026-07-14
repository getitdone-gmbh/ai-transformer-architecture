/**
 * Mini chat terminal for our own model (Ink = React for the terminal).
 *
 * Prerequisite: the inference server is running →  python chat_server.py
 * Start:                                           npm start
 *
 * The model is a BASE MODEL: it continues text, it does not answer.
 * So phrase prompts as sentence openers ("Die Hauptstadt von Frankreich" —
 * the model only speaks German), not as questions — until the SFT version
 * arrives. ;)
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

	// Fetch model info on startup — doubles as a server health check.
	useEffect(() => {
		fetch(`${SERVER}/info`)
			.then((r) => r.json())
			.then(setInfo)
			.catch(() => setInfoError(
				`No server at ${SERVER} — start it first: python chat_server.py`,
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
						? `Your model · ${mio}M parameters · ${info.device} · ${info.checkpoint}`
						: infoError ?? 'Connecting to server...'}
				</Text>
			</Box>

			{history.map((entry, i) => (
				<Box key={i} flexDirection="column" marginTop={1}>
					<Text color="yellow" bold>{'You    › '}<Text color="white">{entry.prompt}</Text></Text>
					{entry.error
						? <Text color="red">Error: {entry.error}</Text>
						: (
							<>
								<Text color="green" bold>{'Model  › '}<Text color="white">{entry.text}</Text></Text>
								<Text dimColor>         {entry.stats}</Text>
							</>
						)}
				</Box>
			))}

			<Box marginTop={1}>
				{busy ? (
					<Text color="green"><Spinner type="dots" /> generating...</Text>
				) : (
					<>
						<Text color="yellow" bold>{'You    › '}</Text>
						<TextInput
							value={input}
							onChange={setInput}
							onSubmit={submit}
							placeholder="Type a sentence opener in German (Enter sends, Esc quits)"
						/>
					</>
				)}
			</Box>
		</Box>
	);
}

if (!process.stdin.isTTY) {
	console.error('Needs a real terminal (TTY) — please run it directly from the shell.');
	process.exit(1);
}
render(<App />);
