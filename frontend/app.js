// 🇻🇳 TROLYLUAT.VN EMERALD & GOLD FRONTEND JACK-IN SCRIPT

document.addEventListener("DOMContentLoaded", () => {
    const topKSlider = document.getElementById("topKSlider");
    const topKValue = document.getElementById("topKValue");
    const btnNewChat = document.getElementById("btnNewChat");
    const btnClearChat = document.getElementById("btnClearChat");
    const heroSection = document.getElementById("heroSection");
    const chatMessagesList = document.getElementById("chatMessagesList");
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const btnSend = document.getElementById("btnSend");

    // Auto-expand textarea
    userInput.addEventListener("input", () => {
        userInput.style.height = "auto";
        userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
    });

    // Sync slider value
    topKSlider.addEventListener("input", (e) => {
        topKValue.textContent = e.target.value;
    });

    // New Chat / Reset
    btnNewChat.addEventListener("click", resetChat);
    btnClearChat.addEventListener("click", resetChat);

    function resetChat() {
        chatMessagesList.innerHTML = "";
        heroSection.style.display = "block";
        userInput.value = "";
        userInput.style.height = "auto";
    }

    // Quick tag & Suggestion item click handlers
    document.querySelectorAll(".quick-tag-btn, .suggestion-item").forEach((el) => {
        el.addEventListener("click", () => {
            const query = el.getAttribute("data-query");
            if (query) {
                userInput.value = query;
                submitQuery(query);
            }
        });
    });

    // Form Submit
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (query) {
            submitQuery(query);
        }
    });

    // Enter Key Handler (Shift+Enter for newline)
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    async function submitQuery(query) {
        if (heroSection.style.display !== "none") {
            heroSection.style.display = "none";
        }

        // Render User Message
        appendMessage("user", query);

        // Reset Input
        userInput.value = "";
        userInput.style.height = "auto";
        btnSend.disabled = true;

        // Loading message
        const loadingId = appendLoadingMessage();
        const topK = parseInt(topKSlider.value, 10) || 5;

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query, top_k: topK })
            });

            removeLoadingMessage(loadingId);

            if (!response.ok) {
                const errData = await response.json();
                appendMessage("assistant", `❌ **Lỗi máy chủ:** ${errData.detail || "Không thể kết nối API."}`);
                return;
            }

            const data = await response.json();
            const answer = data.answer || "Chưa có câu trả lời.";
            const sources = data.sources || [];
            const retrievalSrc = data.retrieval_source || "hybrid";

            appendMessage("assistant", answer, sources, retrievalSrc);

        } catch (error) {
            removeLoadingMessage(loadingId);
            appendMessage("assistant", `❌ **Lỗi kết nối:** Không thể gửi yêu cầu tới FastAPI Server (${error.message}).`);
        } finally {
            btnSend.disabled = false;
        }
    }

    function appendMessage(role, content, sources = [], retrievalSrc = "hybrid") {
        const msgDiv = document.createElement("div");
        msgDiv.className = `chat-msg ${role}`;

        const avatarDiv = document.createElement("div");
        avatarDiv.className = "msg-avatar";
        avatarDiv.textContent = role === "user" ? "👤" : "⚖️";

        const wrapperDiv = document.createElement("div");
        wrapperDiv.className = "msg-content-wrapper";

        // Render Answer Bubble
        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "msg-bubble";

        let formattedContent = escapeHtml(content)
            .replace(/\n/g, "<br>")
            .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")
            .replace(/`(.*?)`/g, "<code>$1</code>");

        bubbleDiv.innerHTML = `<p>${formattedContent}</p>`;
        wrapperDiv.appendChild(bubbleDiv);

        // Render TrolyLuat style Sources Card
        if (role === "assistant" && sources && sources.length > 0) {
            const sourcesCard = document.createElement("div");
            sourcesCard.className = "sources-card";

            let sourcesHtml = `
                <div class="sources-title">
                    <span>🏛️ CĂN CỨ PHÁP LÝ & TRÍCH DẪN (${sources.length} đoạn | Cơ chế: ${retrievalSrc.toUpperCase()})</span>
                    <span style="font-size:0.75rem; color:var(--brand-green); font-weight:700;">✔ 100% Chính Chủ</span>
                </div>
                <div class="sources-grid">
            `;

            sources.forEach((src, idx) => {
                const meta = src.metadata || {};
                const sourceName = meta.source || "Văn bản pháp luật";
                const docType = meta.type || "legal";
                const score = src.score ? src.score.toFixed(4) : "0.0000";

                let badgeClass = "badge-news";
                let badgeText = `📄 ${docType.toUpperCase()}`;

                const sLower = sourceName.toLowerCase();
                if (sLower.includes("luat") || sLower.includes("bo-luat")) {
                    badgeClass = "badge-blld";
                    badgeText = "📜 BỘ LUẬT LAO ĐỘNG 2019";
                } else if (sLower.includes("nghi_dinh") || sLower.includes("74-cp")) {
                    badgeClass = "badge-nghidinh";
                    badgeText = "📑 NGHỊ ĐỊNH HƯỚNG DẪN";
                }

                let preview = src.content || "";
                if (preview.length > 220) {
                    preview = preview.substring(0, 220) + "...";
                }

                sourcesHtml += `
                    <div class="source-item">
                        <span class="${badgeClass}">${badgeText}</span>
                        <div class="source-title">[${idx + 1}] ${escapeHtml(sourceName)} (Score: ${score})</div>
                        <div class="source-snippet">"${escapeHtml(preview)}"</div>
                    </div>
                `;
            });

            sourcesHtml += `</div>`;
            sourcesCard.innerHTML = sourcesHtml;
            wrapperDiv.appendChild(sourcesCard);
        }

        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(wrapperDiv);

        chatMessagesList.appendChild(msgDiv);

        // Auto Scroll
        const viewport = document.getElementById("chatViewport");
        viewport.scrollTop = viewport.scrollHeight;
    }

    function appendLoadingMessage() {
        const loadingId = "loading_" + Date.now();
        const msgDiv = document.createElement("div");
        msgDiv.className = "chat-msg assistant";
        msgDiv.id = loadingId;

        const avatarDiv = document.createElement("div");
        avatarDiv.className = "msg-avatar";
        avatarDiv.textContent = "⚖️";

        const wrapperDiv = document.createElement("div");
        wrapperDiv.className = "msg-content-wrapper";

        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "msg-bubble";
        bubbleDiv.innerHTML = `<p>⏳ <i>Đang tra cứu điều khoản Bộ luật Lao động & tổng hợp câu trả lời...</i></p>`;

        wrapperDiv.appendChild(bubbleDiv);
        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(wrapperDiv);

        chatMessagesList.appendChild(msgDiv);

        const viewport = document.getElementById("chatViewport");
        viewport.scrollTop = viewport.scrollHeight;

        return loadingId;
    }

    function removeLoadingMessage(loadingId) {
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) {
            loadingEl.remove();
        }
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
