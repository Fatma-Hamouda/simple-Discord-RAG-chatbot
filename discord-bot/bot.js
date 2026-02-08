/**
 * Discord RAG Bot - Main Bot File
 * Handles Discord interactions and communicates with the backend API
 */

const { Client, GatewayIntentBits, EmbedBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle } = require('discord.js');
const axios = require('axios');
require('dotenv').config();

// Configuration
const DISCORD_TOKEN = process.env.DISCORD_TOKEN;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';
const BOT_PREFIX = process.env.BOT_PREFIX || '!ask';

// Initialize Discord client
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
        GatewayIntentBits.DirectMessages
    ]
});

// Logging
const log = {
    info: (msg) => console.log(`[INFO] ${new Date().toISOString()} - ${msg}`),
    error: (msg) => console.error(`[ERROR] ${new Date().toISOString()} - ${msg}`),
    warn: (msg) => console.warn(`[WARN] ${new Date().toISOString()} - ${msg}`)
};

// Bot statistics
const stats = {
    queries: 0,
    errors: 0,
    feedbackReceived: 0,
    startTime: new Date()
};

/**
 * Query the RAG backend
 */
async function queryRAG(question, userId) {
    try {
        const response = await axios.post(`${BACKEND_URL}/api/query`, {
            question: question,
            top_k: 3,
            user_id: userId
        }, {
            timeout: 30000 // 30 second timeout
        });

        return response.data;
    } catch (error) {
        log.error(`RAG query error: ${error.message}`);
        throw new Error('Failed to get answer from RAG system');
    }
}

/**
 * Send feedback to backend
 */
async function sendFeedback(query, answer, feedback, userId, comment = null) {
    try {
        await axios.post(`${BACKEND_URL}/api/feedback`, {
            query: query,
            answer: answer,
            feedback: feedback,
            user_id: userId,
            comment: comment
        });
        log.info(`Feedback sent: ${feedback} from ${userId}`);
    } catch (error) {
        log.error(`Failed to send feedback: ${error.message}`);
    }
}

/**
 * Create an embed for the bot response
 */
function createResponseEmbed(question, answer, sources, color = 0x0099FF) {
    const embed = new EmbedBuilder()
        .setColor(color)
        .setTitle('🤖 AI Bootcamp Assistant')
        .setDescription(answer)
        .addFields({
            name: '❓ Question',
            value: question.substring(0, 1000),
            inline: false
        })
        .setTimestamp()
        .setFooter({ text: 'Discord RAG Bot | React with 👍 or 👎 to provide feedback' });

    return embed;
}

/**
 * Create an error embed
 */
function createErrorEmbed(error) {
    return new EmbedBuilder()
        .setColor(0xFF0000)
        .setTitle('❌ Error')
        .setDescription(`I encountered an error while processing your request:\n\`\`\`${error}\`\`\``)
        .addFields({
            name: '💡 Troubleshooting',
            value: '• Make sure the backend API is running\n• Check if documents are ingested\n• Try again in a moment'
        })
        .setTimestamp();
}

/**
 * Handle bot mentions or prefix commands
 */
async function handleQuery(message) {
    // Extract the question
    let question = message.content;
    
    // Remove bot mention if present
    question = question.replace(/<@!?\d+>/g, '').trim();
    
    // Remove prefix if present
    if (question.startsWith(BOT_PREFIX)) {
        question = question.substring(BOT_PREFIX.length).trim();
    }

    // Validate question
    if (!question || question.length < 3) {
        await message.reply('Please ask a valid question about the AI Bootcamp!');
        return;
    }

    log.info(`Query from ${message.author.tag}: ${question}`);
    stats.queries++;

    // Show typing indicator
    await message.channel.sendTyping();

    try {
        // Query the RAG system
        const result = await queryRAG(question, message.author.id);

        // Create and send response embed
        const embed = createResponseEmbed(
            question,
            result.answer,
            result.sources
        );

        const reply = await message.reply({ embeds: [embed] });

        // Add reaction buttons for feedback
        await reply.react('👍');
        await reply.react('👎');

        // Create reaction collector
        const filter = (reaction, user) => {
            return ['👍', '👎'].includes(reaction.emoji.name) && user.id === message.author.id;
        };

        const collector = reply.createReactionCollector({ filter, time: 300000, max: 1 }); // 5 minutes

        collector.on('collect', async (reaction, user) => {
            const feedback = reaction.emoji.name === '👍' ? 'helpful' : 'not_helpful';
            await sendFeedback(question, result.answer, feedback, user.id);
            stats.feedbackReceived++;
            
            // Send confirmation
            const confirmMsg = feedback === 'helpful' 
                ? '✅ Thanks for the positive feedback!' 
                : '📝 Thanks for the feedback! We\'ll work on improving.';
            
            await message.channel.send(confirmMsg);
        });

    } catch (error) {
        log.error(`Error handling query: ${error.message}`);
        stats.errors++;
        
        const errorEmbed = createErrorEmbed(error.message);
        await message.reply({ embeds: [errorEmbed] });
    }
}

/**
 * Bot ready event
 */
client.on('ready', () => {
    log.info(`✓ Bot logged in as ${client.user.tag}`);
    log.info(`✓ Serving ${client.guilds.cache.size} guilds`);
    log.info(`✓ Backend URL: ${BACKEND_URL}`);
    log.info(`✓ Command prefix: ${BOT_PREFIX}`);
    
    client.user.setActivity('AI Bootcamp Q&A | !ask <question>', { type: 'PLAYING' });
    
    // Test backend connection
    axios.get(`${BACKEND_URL}/health`)
        .then(response => {
            log.info('✓ Backend connection successful');
        })
        .catch(error => {
            log.error('✗ Backend connection failed! Make sure the backend is running.');
        });
});

/**
 * Message handler
 */
client.on('messageCreate', async (message) => {
    // Ignore bot messages
    if (message.author.bot) return;

    // Check if bot is mentioned or message starts with prefix
    const isMentioned = message.mentions.has(client.user);
    const hasPrefix = message.content.startsWith(BOT_PREFIX);

    if (isMentioned || hasPrefix) {
        await handleQuery(message);
    }
});

/**
 * Handle help command
 */
client.on('messageCreate', async (message) => {
    if (message.content === '!help' && !message.author.bot) {
        const helpEmbed = new EmbedBuilder()
            .setColor(0x0099FF)
            .setTitle('🤖 Discord RAG Bot Help')
            .setDescription('I can answer questions about the AI Engineering Bootcamp!')
            .addFields(
                {
                    name: '📖 How to Use',
                    value: `• Mention me: @${client.user.tag} What is RAG?\n• Use prefix: ${BOT_PREFIX} What are the project phases?\n• DM me directly with your question`
                },
                {
                    name: '👍 Feedback',
                    value: 'React with 👍 if my answer was helpful, or 👎 if it wasn\'t. This helps me improve!'
                },
                {
                    name: '📊 Stats',
                    value: `Queries answered: ${stats.queries}\nFeedback received: ${stats.feedbackReceived}`
                }
            )
            .setTimestamp();

        await message.reply({ embeds: [helpEmbed] });
    }
});

/**
 * Error handling
 */
client.on('error', error => {
    log.error(`Discord client error: ${error.message}`);
});

process.on('unhandledRejection', error => {
    log.error(`Unhandled promise rejection: ${error}`);
});

// Start the bot
if (!DISCORD_TOKEN) {
    log.error('DISCORD_TOKEN not found in environment variables!');
    process.exit(1);
}

client.login(DISCORD_TOKEN)
    .then(() => {
        log.info('Discord bot starting...');
    })
    .catch(error => {
        log.error(`Failed to login: ${error.message}`);
        process.exit(1);
    });

// Graceful shutdown
process.on('SIGINT', () => {
    log.info('Shutting down gracefully...');
    client.destroy();
    process.exit(0);
});
